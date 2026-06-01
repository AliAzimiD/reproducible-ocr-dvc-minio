from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


VALID_ANGLES = (0, 90, 180, 270)


@dataclass(frozen=True)
class OrientationPrediction:
    image_path: str
    upright_path: str
    predicted_angle: int
    confidence: float
    backend: str


def repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def reset_output_dir(output_dir: Path) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    upright_dir = output_dir / "upright"
    upright_dir.mkdir(parents=True, exist_ok=True)
    return upright_dir


def load_manifest(input_dir: Path) -> list[dict]:
    manifest_path = input_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing extraction manifest: {manifest_path}")

    rows: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mock_predict(image_path: Path) -> tuple[int, float]:
    digest = hashlib.sha256(image_path.as_posix().encode("utf-8")).digest()
    angle = VALID_ANGLES[digest[0] % len(VALID_ANGLES)]
    confidence = round(0.75 + (digest[1] / 255) * 0.24, 4)
    return angle, confidence


class PaddleOrientationPredictor:
    def __init__(self, model_dir: Path) -> None:
        if not model_dir.exists():
            raise FileNotFoundError(
                f"ONNX orientation backend selected, but model directory is missing: {model_dir}. "
                "Add the model artifact with DVC or use --backend mock."
            )
        self.model_path = model_dir / "PP-LCNet_x1_0_doc_ori_infer.onnx"
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model file is missing: {self.model_path}. "
                "Expected PP-LCNet_x1_0_doc_ori_infer.onnx inside --model-dir."
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "ONNX orientation backend requires onnxruntime. "
                "Install requirements/paddle.txt before using --backend paddle."
            ) from exc

        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.labels = VALID_ANGLES

    def predict(self, image_path: Path) -> tuple[int, float]:
        input_data = self._preprocess(image_path)
        output = self.session.run(None, {self.input_name: input_data})
        logits = np.asarray(output[0]).reshape(-1)
        probabilities = self._softmax(logits)
        class_id = int(np.argmax(probabilities))
        if class_id >= len(self.labels):
            raise ValueError(f"ONNX model returned unsupported class index: {class_id}")
        return self.labels[class_id], float(probabilities[class_id])

    @staticmethod
    def _preprocess(image_path: Path) -> np.ndarray:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image dimensions for {image_path}: {image.size}")

            resize_short = 256
            if width < height:
                new_width = resize_short
                new_height = round(height * resize_short / width)
            else:
                new_height = resize_short
                new_width = round(width * resize_short / height)
            image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)

            crop_size = 224
            left = (new_width - crop_size) // 2
            top = (new_height - crop_size) // 2
            image = image.crop((left, top, left + crop_size, top + crop_size))

            array = np.asarray(image).astype("float32") / 255.0
            mean = np.asarray([0.485, 0.456, 0.406], dtype="float32")
            std = np.asarray([0.229, 0.224, 0.225], dtype="float32")
            array = (array - mean) / std
            array = np.transpose(array, (2, 0, 1))
            return np.expand_dims(array, axis=0).astype("float32")

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        return exp / np.sum(exp)


def rotate_to_upright(image_path: Path, output_path: Path, predicted_angle: int) -> None:
    if predicted_angle not in VALID_ANGLES:
        raise ValueError(f"Unsupported angle {predicted_angle}; expected one of {VALID_ANGLES}")
    with Image.open(image_path) as image:
        image.convert("RGB").rotate(-predicted_angle, expand=True).save(output_path, format="PNG")


def write_outputs(output_dir: Path, predictions: Iterable[OrientationPrediction]) -> None:
    predictions = list(predictions)
    manifest_path = output_dir / "manifest.jsonl"
    preds_path = output_dir / "preds.csv"

    with manifest_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(asdict(prediction), sort_keys=True) + "\n")

    with preds_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "upright_path", "predicted_angle", "confidence", "backend"],
        )
        writer.writeheader()
        for prediction in predictions:
            writer.writerow(asdict(prediction))


def orient_all(input_dir: Path, output_dir: Path, model_dir: Path, backend: str, repo_root: Path) -> list[OrientationPrediction]:
    if backend not in {"mock", "paddle"}:
        raise ValueError("backend must be either 'mock' or 'paddle'")

    rows = load_manifest(input_dir)
    upright_dir = reset_output_dir(output_dir)
    paddle_predictor = PaddleOrientationPredictor(model_dir) if backend == "paddle" else None

    predictions: list[OrientationPrediction] = []
    for row in rows:
        image_path = repo_root / row["image_path"]
        upright_path = upright_dir / Path(row["image_path"]).name
        if paddle_predictor is None:
            angle, confidence = mock_predict(image_path)
        else:
            angle, confidence = paddle_predictor.predict(image_path)

        rotate_to_upright(image_path, upright_path, angle)
        predictions.append(
            OrientationPrediction(
                image_path=repo_relative(image_path, repo_root),
                upright_path=repo_relative(upright_path, repo_root),
                predicted_angle=angle,
                confidence=confidence,
                backend=backend,
            )
        )

    write_outputs(output_dir, predictions)
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify and rotate document images into upright orientation.")
    parser.add_argument("--input", required=True, type=Path, help="Directory containing extracted images and manifest.jsonl.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for orientation artifacts.")
    parser.add_argument("--model-dir", required=True, type=Path, help="Paddle orientation model directory.")
    parser.add_argument("--backend", choices=["mock", "paddle"], default="mock", help="Orientation backend.")
    parser.add_argument("--repo-root", default=Path.cwd(), type=Path, help="Repository root for relative manifest paths.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = orient_all(args.input, args.output, args.model_dir, args.backend, args.repo_root)
    print(f"Oriented {len(predictions)} image(s) into {args.output}")


if __name__ == "__main__":
    main()
