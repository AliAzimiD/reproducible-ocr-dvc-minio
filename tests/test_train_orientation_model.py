from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.stages.train_orientation_model import discover_source_images, main, split_sources


def test_discover_source_images_recurses_supported_formats(tmp_path: Path) -> None:
    source_dir = tmp_path / "sources"
    nested = source_dir / "nested"
    nested.mkdir(parents=True)
    Image.new("RGB", (12, 8), "white").save(source_dir / "a.jpg")
    Image.new("RGB", (12, 8), "white").save(nested / "b.png")
    (source_dir / "ignore.txt").write_text("nope", encoding="utf-8")

    images = discover_source_images(source_dir)

    assert [path.name for path in images] == ["a.jpg", "b.png"]


def test_split_sources_is_reproducible(tmp_path: Path) -> None:
    images = [tmp_path / f"{index}.jpg" for index in range(10)]

    first = split_sources(images, val_fraction=0.2, seed=7)
    second = split_sources(images, val_fraction=0.2, seed=7)

    assert first == second
    assert len(first[0]) == 8
    assert len(first[1]) == 2
    assert not set(first[0]) & set(first[1])


def test_train_stage_dry_run_writes_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_dir = tmp_path / "sources"
    output_dir = tmp_path / "model"
    source_dir.mkdir()
    Image.new("RGB", (16, 12), "white").save(source_dir / "a.jpg")
    Image.new("RGB", (16, 12), "white").save(source_dir / "b.jpg")

    monkeypatch.setattr(
        "sys.argv",
        [
            "train_orientation_model.py",
            "--source-dir",
            str(source_dir),
            "--output",
            str(output_dir),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--pretrained",
            "false",
            "--dry-run",
            "--repo-root",
            str(tmp_path),
        ],
    )

    main()

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["dry_run"] is True
    assert metrics["train_samples"] == 4
    assert metrics["val_samples"] == 4
    assert (output_dir / "train_val_split.json").exists()
    assert (output_dir / "label_map.csv").exists()
