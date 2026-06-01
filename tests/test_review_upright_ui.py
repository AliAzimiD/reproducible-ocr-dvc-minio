from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.tools.review_upright_ui import ReviewApp, discover_images, rotate_and_save
import src.stages.orient_and_rotate as orient_module


def test_discover_images_ignores_non_images(tmp_path: Path) -> None:
    input_dir = tmp_path / "images_flat"
    input_dir.mkdir()
    Image.new("RGB", (20, 10), "white").save(input_dir / "a.png")
    (input_dir / "manifest.jsonl").write_text("{}", encoding="utf-8")

    images = discover_images(input_dir)

    assert [path.name for path in images] == ["a.png"]


def test_rotate_and_save_swaps_dimensions(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    Image.new("RGB", (30, 10), "white").save(source)

    rotate_and_save(source, output, 90)

    assert Image.open(output).size == (10, 30)


def test_review_app_saves_current_and_manifest(tmp_path: Path) -> None:
    input_dir = tmp_path / "data" / "interim" / "images_flat"
    output_dir = tmp_path / "data" / "interim" / "images_flat_corrected_upright"
    input_dir.mkdir(parents=True)
    Image.new("RGB", (30, 10), "white").save(input_dir / "a.png")

    app = ReviewApp(input_dir, output_dir, repo_root=tmp_path)
    preview_status = app.preview_current("rotate_right")
    preview_bytes, preview_mime = app.current_image_bytes()
    status = app.review_current("save_preview")

    output = output_dir / "a.png"
    assert preview_status["preview_rotation"] == 90
    assert preview_mime == "image/png"
    assert preview_bytes
    assert output.exists()
    assert Image.open(output).size == (10, 30)
    assert status["done"] is True

    rows = [
        json.loads(line)
        for line in (output_dir / "review_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["source_path"] == "data/interim/images_flat/a.png"
    assert rows[0]["output_path"] == "data/interim/images_flat_corrected_upright/a.png"
    assert rows[0]["action"] == "save_preview"
    assert rows[0]["rotation_degrees"] == 90


def test_review_app_predicts_and_orders_non_zero_then_low_confidence(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "data" / "interim" / "images_flat"
    output_dir = tmp_path / "data" / "interim" / "images_flat_corrected_upright"
    model_dir = tmp_path / "model"
    input_dir.mkdir(parents=True)
    model_dir.mkdir()
    for name in ["a.png", "b.png", "c.png", "d.png"]:
        Image.new("RGB", (16, 12), "white").save(input_dir / name)

    class FakePredictor:
        def __init__(self, model_dir: Path) -> None:
            self.model_dir = model_dir

        def predict(self, image_path: Path) -> tuple[int, float]:
            return {
                "a.png": (0, 0.9),
                "b.png": (90, 0.8),
                "c.png": (0, 0.2),
                "d.png": (180, 0.7),
            }[image_path.name]

    monkeypatch.setattr(orient_module, "PaddleOrientationPredictor", FakePredictor)

    app = ReviewApp(input_dir, output_dir, repo_root=tmp_path, model_dir=model_dir)

    assert [path.name for path in app.images] == ["d.png", "b.png", "c.png", "a.png"]
    assert (output_dir / "predictions.csv").exists()
    first_status = app.status()
    assert first_status["source_path"].endswith("d.png")
    assert first_status["predicted_angle"] == 180
    assert first_status["prediction_confidence"] == 0.7
