from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.stages.orient_and_rotate import PaddleOrientationPredictor, mock_predict, orient_all, rotate_to_upright


def test_rotate_to_upright_preserves_and_swaps_dimensions(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (40, 20), "white").save(image_path)

    output_0 = tmp_path / "angle-0.png"
    output_90 = tmp_path / "angle-90.png"
    output_180 = tmp_path / "angle-180.png"
    output_270 = tmp_path / "angle-270.png"

    rotate_to_upright(image_path, output_0, 0)
    rotate_to_upright(image_path, output_90, 90)
    rotate_to_upright(image_path, output_180, 180)
    rotate_to_upright(image_path, output_270, 270)

    assert Image.open(output_0).size == (40, 20)
    assert Image.open(output_90).size == (20, 40)
    assert Image.open(output_180).size == (40, 20)
    assert Image.open(output_270).size == (20, 40)


def test_mock_predict_is_deterministic(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"not actually read")

    assert mock_predict(image_path) == mock_predict(image_path)
    assert mock_predict(image_path)[0] in {0, 90, 180, 270}


def test_onnx_preprocess_shape_and_softmax(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (320, 180), "white").save(image_path)

    input_data = PaddleOrientationPredictor._preprocess(image_path)
    probabilities = PaddleOrientationPredictor._softmax(np.asarray([0.0, 1.0, 2.0, 3.0], dtype="float32"))

    assert input_data.shape == (1, 3, 224, 224)
    assert input_data.dtype == np.float32
    assert np.isclose(probabilities.sum(), 1.0)
    assert int(np.argmax(probabilities)) == 3


def test_orient_all_writes_manifest_and_predictions(tmp_path: Path) -> None:
    extracted_dir = tmp_path / "data" / "interim" / "images_flat"
    output_dir = tmp_path / "data" / "interim" / "orientation"
    extracted_dir.mkdir(parents=True)
    image_path = extracted_dir / "abc.png"
    Image.new("RGB", (24, 12), "white").save(image_path)
    manifest = {
        "source_path": "data/raw/mixed/sample.png",
        "source_type": "image",
        "page_index": 0,
        "image_path": "data/interim/images_flat/abc.png",
        "width": 24,
        "height": 12,
        "source_sha256": "x",
        "item_hash": "abc",
    }
    (extracted_dir / "manifest.jsonl").write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    predictions = orient_all(extracted_dir, output_dir, tmp_path / "models" / "missing", "mock", tmp_path)

    assert len(predictions) == 1
    assert (output_dir / "manifest.jsonl").exists()
    assert (output_dir / "preds.csv").exists()
    assert (tmp_path / predictions[0].upright_path).exists()

    with (output_dir / "preds.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["backend"] == "mock"
    assert int(rows[0]["predicted_angle"]) in {0, 90, 180, 270}
