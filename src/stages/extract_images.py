from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageSequence


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PDF_SUFFIXES = {".pdf"}


@dataclass(frozen=True)
class ExtractedImage:
    source_path: str
    source_type: str
    page_index: int
    image_path: str
    width: int
    height: int
    source_sha256: str
    item_hash: str


def repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_stem(source: Path, page_index: int, source_hash: str, repo_root: Path) -> str:
    payload = f"{repo_relative(source, repo_root)}\n{page_index}\n{source_hash}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def iter_input_files(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES | PDF_SUFFIXES:
            yield path


def save_pil_image(image: Image.Image, output_path: Path) -> tuple[int, int]:
    normalized = image.convert("RGB")
    normalized.save(output_path, format="PNG")
    return normalized.size


def extract_image_file(source: Path, output_dir: Path, repo_root: Path) -> list[ExtractedImage]:
    source_hash = file_sha256(source)
    records: list[ExtractedImage] = []
    with Image.open(source) as image:
        for index, frame in enumerate(ImageSequence.Iterator(image)):
            item_hash = deterministic_stem(source, index, source_hash, repo_root)
            output_path = output_dir / f"{item_hash}.png"
            width, height = save_pil_image(frame, output_path)
            records.append(
                ExtractedImage(
                    source_path=repo_relative(source, repo_root),
                    source_type="image",
                    page_index=index,
                    image_path=repo_relative(output_path, repo_root),
                    width=width,
                    height=height,
                    source_sha256=source_hash,
                    item_hash=item_hash,
                )
            )
    return records


def extract_pdf_file(source: Path, output_dir: Path, repo_root: Path) -> list[ExtractedImage]:
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PDF extraction requires PyMuPDF. Install requirements.txt first.") from exc

    source_hash = file_sha256(source)
    records: list[ExtractedImage] = []
    document = fitz.open(source)
    try:
        for index, page in enumerate(document):
            item_hash = deterministic_stem(source, index, source_hash, repo_root)
            output_path = output_dir / f"{item_hash}.png"
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            pixmap.save(output_path)
            records.append(
                ExtractedImage(
                    source_path=repo_relative(source, repo_root),
                    source_type="pdf",
                    page_index=index,
                    image_path=repo_relative(output_path, repo_root),
                    width=pixmap.width,
                    height=pixmap.height,
                    source_sha256=source_hash,
                    item_hash=item_hash,
                )
            )
    finally:
        document.close()
    return records


def extract_all(input_dir: Path, output_dir: Path, repo_root: Path) -> list[ExtractedImage]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    reset_output_dir(output_dir)
    records: list[ExtractedImage] = []
    for source in iter_input_files(input_dir):
        suffix = source.suffix.lower()
        if suffix in PDF_SUFFIXES:
            records.extend(extract_pdf_file(source, output_dir, repo_root))
        elif suffix in IMAGE_SUFFIXES:
            records.extend(extract_image_file(source, output_dir, repo_root))

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract PDFs and image files into a flat image dataset.")
    parser.add_argument("--input", required=True, type=Path, help="Directory containing mixed raw inputs.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for extracted PNG images.")
    parser.add_argument("--repo-root", default=Path.cwd(), type=Path, help="Repository root for relative manifest paths.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = extract_all(args.input, args.output, args.repo_root)
    print(f"Extracted {len(records)} image(s) into {args.output}")


if __name__ == "__main__":
    main()
