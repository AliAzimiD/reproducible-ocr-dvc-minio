from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.stages.extract_images import deterministic_stem, extract_all, file_sha256


def test_deterministic_stem_is_stable(tmp_path: Path) -> None:
    source = tmp_path / "raw" / "doc.png"
    source.parent.mkdir()
    source.write_bytes(b"image-ish")
    source_hash = file_sha256(source)

    first = deterministic_stem(source, 0, source_hash, tmp_path)
    second = deterministic_stem(source, 0, source_hash, tmp_path)
    different_page = deterministic_stem(source, 1, source_hash, tmp_path)

    assert first == second
    assert first != different_page
    assert len(first) == 24


def test_extracts_common_image_and_manifest(tmp_path: Path) -> None:
    input_dir = tmp_path / "data" / "raw" / "mixed"
    output_dir = tmp_path / "data" / "interim" / "images_flat"
    input_dir.mkdir(parents=True)
    Image.new("RGB", (32, 20), "white").save(input_dir / "sample.jpg")

    records = extract_all(input_dir, output_dir, tmp_path)
    manifest_rows = [
        json.loads(line)
        for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(records) == 1
    assert len(manifest_rows) == 1
    assert manifest_rows[0]["source_path"] == "data/raw/mixed/sample.jpg"
    assert manifest_rows[0]["source_type"] == "image"
    assert manifest_rows[0]["page_index"] == 0
    assert manifest_rows[0]["width"] == 32
    assert manifest_rows[0]["height"] == 20
    assert (tmp_path / manifest_rows[0]["image_path"]).exists()


def test_extracts_pdf_pages_when_pymupdf_available(tmp_path: Path) -> None:
    pytest.importorskip("fitz")

    input_dir = tmp_path / "data" / "raw" / "mixed"
    output_dir = tmp_path / "data" / "interim" / "images_flat"
    input_dir.mkdir(parents=True)
    Image.new("RGB", (30, 18), "white").save(input_dir / "one-page.pdf", "PDF")

    records = extract_all(input_dir, output_dir, tmp_path)

    assert len(records) == 1
    assert records[0].source_type == "pdf"
    assert records[0].page_index == 0
    assert Path(tmp_path / records[0].image_path).exists()
