from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
ROTATION_BY_CLASS = {0: 0, 1: 90, 2: 180, 3: 270}


@dataclass(frozen=True)
class SplitSummary:
    seed: int
    train_sources: int
    val_sources: int
    train_samples: int
    val_samples: int


def repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def discover_source_images(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Training source directory does not exist: {source_dir}")
    images = [
        path
        for path in sorted(source_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if len(images) < 2:
        raise ValueError(f"At least 2 source images are required for training; found {len(images)} in {source_dir}")
    return images


def split_sources(images: list[Path], val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    if not 0 < val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * val_fraction)))
    return shuffled[val_count:], shuffled[:val_count]


def write_split(output_dir: Path, train_images: Iterable[Path], val_images: Iterable[Path], seed: int, repo_root: Path) -> SplitSummary:
    train_images = list(train_images)
    val_images = list(val_images)
    split = {
        "seed": seed,
        "train_images": [repo_relative(path, repo_root) for path in train_images],
        "val_images": [repo_relative(path, repo_root) for path in val_images],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_val_split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    return SplitSummary(
        seed=seed,
        train_sources=len(train_images),
        val_sources=len(val_images),
        train_samples=len(train_images) * len(ROTATION_BY_CLASS),
        val_samples=len(val_images) * len(ROTATION_BY_CLASS),
    )


def write_dry_run_outputs(output_dir: Path, split_summary: SplitSummary, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "dry_run": True,
        "train_accuracy": None,
        "val_accuracy": None,
        "train_loss": None,
        "val_loss": None,
        **asdict(split_summary),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "training_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    (output_dir / "label_map.json").write_text(json.dumps(ROTATION_BY_CLASS, indent=2), encoding="utf-8")


def write_skipped_outputs(output_dir: Path, split_summary: SplitSummary, args: argparse.Namespace, reason: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "skipped": True,
        "skip_reason": reason,
        "train_accuracy": None,
        "val_accuracy": None,
        "train_loss": None,
        "val_loss": None,
        **asdict(split_summary),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "training_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    (output_dir / "label_map.json").write_text(json.dumps(ROTATION_BY_CLASS, indent=2), encoding="utf-8")


def import_torch_stack():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models, transforms
    except ImportError as exc:
        raise RuntimeError(
            "Training requires PyTorch and torchvision. Install the optional training dependencies first: "
            "pip install torch torchvision tqdm tensorboard"
        ) from exc
    return torch, nn, optim, DataLoader, Dataset, models, transforms


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in {"RGB", "L"}:
            return image.convert("RGB")
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba)
        return background


def rotate_image(image: Image.Image, angle: int) -> Image.Image:
    if angle == 0:
        return image.copy()
    if angle == 90:
        return image.transpose(Image.Transpose.ROTATE_90)
    if angle == 180:
        return image.transpose(Image.Transpose.ROTATE_180)
    if angle == 270:
        return image.transpose(Image.Transpose.ROTATE_270)
    raise ValueError(f"Unsupported rotation angle: {angle}")


def build_model(models, nn, pretrained: bool, num_blocks_to_unfreeze: int):
    weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_v2_s(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    if num_blocks_to_unfreeze > 0:
        for block in model.features[-num_blocks_to_unfreeze:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
    input_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True), nn.Linear(input_features, len(ROTATION_BY_CLASS)))
    return model


def load_pretrained_checkpoint(torch, model, checkpoint_path: Path, device) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint does not exist: {checkpoint_path}")

    try:
        artifact = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        artifact = torch.load(checkpoint_path, map_location=device)

    if isinstance(artifact, dict) and "model_state_dict" in artifact:
        state_dict = artifact["model_state_dict"]
    elif isinstance(artifact, dict) and "state_dict" in artifact:
        state_dict = artifact["state_dict"]
    elif isinstance(artifact, dict):
        state_dict = artifact
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_key = key.removeprefix("module.").removeprefix("_orig_mod.")
        cleaned_state_dict[cleaned_key] = value

    load_result = model.load_state_dict(cleaned_state_dict, strict=False)
    return {
        "path": str(checkpoint_path),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }


def train_model(args: argparse.Namespace, train_images: list[Path], val_images: list[Path], split_summary: SplitSummary) -> dict:
    torch, nn, optim, DataLoader, Dataset, models, transforms = import_torch_stack()

    class OrientationDataset(Dataset):
        def __init__(self, image_paths: list[Path], transform):
            self.image_paths = image_paths
            self.transform = transform
            self.class_ids = sorted(ROTATION_BY_CLASS)

        def __len__(self) -> int:
            return len(self.image_paths) * len(self.class_ids)

        def __getitem__(self, index: int):
            image_index = index // len(self.class_ids)
            class_id = self.class_ids[index % len(self.class_ids)]
            image = rotate_image(load_rgb_image(self.image_paths[image_index]), ROTATION_BY_CLASS[class_id])
            return self.transform(image), torch.tensor(class_id, dtype=torch.long)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.image_size, scale=(0.85, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size + 32, args.image_size + 32)),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_loader = DataLoader(
        OrientationDataset(train_images, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        OrientationDataset(val_images, val_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(models, nn, pretrained=args.pretrained, num_blocks_to_unfreeze=args.num_blocks_to_unfreeze).to(device)
    checkpoint_info = None
    if args.pretrained_checkpoint:
        checkpoint_info = load_pretrained_checkpoint(torch, model, args.pretrained_checkpoint, device)
        print(json.dumps({"loaded_pretrained_checkpoint": checkpoint_info}, sort_keys=True))
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.min_learning_rate)

    best_val_accuracy = -1.0
    best_metrics: dict = {}
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += inputs.size(0)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.inference_mode():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += inputs.size(0)
        scheduler.step()

        metrics = {
            "epoch": epoch + 1,
            "epochs": args.epochs,
            "device": device.type,
            "train_loss": train_loss / max(train_total, 1),
            "train_accuracy": train_correct / max(train_total, 1),
            "val_loss": val_loss / max(val_total, 1),
            "val_accuracy": val_correct / max(val_total, 1),
            "learning_rate": scheduler.get_last_lr()[0],
            "duration_seconds": time.time() - start_time,
            "pretrained_checkpoint": checkpoint_info,
            **asdict(split_summary),
        }
        print(json.dumps(metrics, sort_keys=True))
        if metrics["val_accuracy"] > best_val_accuracy:
            best_val_accuracy = metrics["val_accuracy"]
            best_metrics = metrics
            torch.save(model.state_dict(), args.output / "best_model.pth")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_accuracy": best_val_accuracy,
                "args": vars(args),
            },
            args.output / "checkpoint.pth",
        )

    return best_metrics


def write_labels_csv(output_dir: Path) -> None:
    with (output_dir / "label_map.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_id", "applied_rotation_degrees"])
        writer.writeheader()
        for class_id, angle in ROTATION_BY_CLASS.items():
            writer.writerow({"class_id": class_id, "applied_rotation_degrees": angle})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an EfficientNetV2-S image orientation model adapted from duartebarbosadev/deep-image-orientation-detection."
    )
    parser.add_argument("--source-dir", type=Path, required=True, help="Directory containing upright source images.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for trained model artifacts.")
    parser.add_argument("--pretrained-checkpoint", type=Path, default=None, help="Optional .pth checkpoint to warm-start model weights.")
    parser.add_argument("--mode", choices=["skip", "dry-run", "train"], default="train")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-blocks-to-unfreeze", type=int, default=2)
    parser.add_argument("--pretrained", type=parse_bool, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate data and write metadata without importing/training PyTorch.")
    parser.add_argument("--allow-missing-torch", action="store_true", help="Write skipped metadata instead of failing when PyTorch is not installed.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    images = discover_source_images(args.source_dir)
    train_images, val_images = split_sources(images, args.val_fraction, args.seed)
    split_summary = write_split(args.output, train_images, val_images, args.seed, args.repo_root)
    write_labels_csv(args.output)
    (args.output / "label_map.json").write_text(json.dumps(ROTATION_BY_CLASS, indent=2), encoding="utf-8")
    (args.output / "training_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    if args.mode == "skip":
        write_skipped_outputs(args.output, split_summary, args, "Training disabled by train_orientation.mode=skip.")
        print("Training skipped because --mode skip was selected.")
        return

    if args.dry_run or args.mode == "dry-run":
        write_dry_run_outputs(args.output, split_summary, args)
        print(f"Dry run complete: {split_summary.train_samples} train samples, {split_summary.val_samples} validation samples.")
        return

    try:
        metrics = train_model(args, train_images, val_images, split_summary)
    except RuntimeError as exc:
        if not args.allow_missing_torch or "requires PyTorch" not in str(exc):
            raise
        write_skipped_outputs(args.output, split_summary, args, str(exc))
        print(f"Training skipped: {exc}")
        return
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Training complete. Best validation accuracy: {metrics.get('val_accuracy')}")


if __name__ == "__main__":
    main()
