from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

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
                f"Paddle backend selected, but model directory is missing: {model_dir}. "
                "Add the model artifact with DVC or use --backend mock."
            )
        try:
            from paddleclas import PaddleClas  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Paddle backend requires PaddleClas and a compatible Paddle runtime. "
                "Install those optional dependencies before using --backend paddle."
            ) from exc

        self._classifier = PaddleClas(model_name=str(model_dir), use_gpu=False)

    def predict(self, image_path: Path) -> tuple[int, float]:
        results = list(self._classifier.predict(input_data=str(image_path)))
        if not results:
            raise RuntimeError(f"Paddle backend returned no prediction for {image_path}")

        result = results[0]
        label = str(result.get("label_names", ["0"])[0])
        score = float(result.get("scores", [0.0])[0])
        angle = int(label)
        if angle not in VALID_ANGLES:
            raise ValueError(f"Unsupported orientation label from Paddle backend: {angle}")
        return angle, score


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
