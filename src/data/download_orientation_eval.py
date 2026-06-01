from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import struct
import time
import zipfile
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import requests
from PIL import Image


GIB = 1024 * 1024 * 1024
OPENITI_RECORD_API = "https://zenodo.org/api/records/19861912"
IDPL_IMAGES_API = "https://api.github.com/repos/FtmsdtHosseini/IDPL-PFOD/contents/images"
KAGGLE_DOWNLOAD_API = "https://www.kaggle.com/api/v1/datasets/download/humansintheloop/arabic-documents-ocr-dataset"
INVOICE_DOWNLOAD_API = "https://www.kaggle.com/api/v1/datasets/download/osamahosamabdellatif/high-quality-invoice-images-for-ocr"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class SourceImage:
    source_dataset: str
    source_id: str
    language: str
    image_path: str
    original_url: str


@dataclass(frozen=True)
class RotatedImage:
    image_path: str
    true_angle: int
    source_dataset: str
    source_id: str
    source_image_path: str
    language: str


def repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def request_with_retries(method: str, url: str, **kwargs) -> requests.Response:
    timeout = kwargs.pop("timeout", 60)
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = requests.request(method, url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def content_length(url: str) -> int:
    response = request_with_retries("HEAD", url, allow_redirects=True)
    return int(response.headers["content-length"])


def ranged_get(url: str, start: int, end: int) -> bytes:
    expected = end - start + 1
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = request_with_retries("GET", url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=90)
            chunks = []
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    chunks.append(chunk)
            data = b"".join(chunks)
            if len(data) != expected:
                raise RuntimeError(f"incomplete range read: expected {expected} bytes, got {len(data)}")
            return data
        except Exception as exc:
            last_error = exc
            time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def parse_zip_central_directory(url: str) -> list[dict]:
    archive_size = content_length(url)
    tail_start = max(0, archive_size - 1024 * 1024)
    tail = ranged_get(url, tail_start, archive_size - 1)

    eocd64_locator_pos = tail.rfind(b"PK\x06\x07")
    if eocd64_locator_pos >= 0:
        _, _, eocd64_offset, _ = struct.unpack_from("<4sLQL", tail, eocd64_locator_pos)
        eocd64 = ranged_get(url, eocd64_offset, eocd64_offset + 80)
        values = struct.unpack_from("<4sQ2H2L4Q", eocd64)
        total_entries = values[6]
        central_size = values[8]
        central_offset = values[9]
    else:
        eocd_pos = tail.rfind(b"PK\x05\x06")
        if eocd_pos < 0:
            raise RuntimeError("Could not find ZIP central directory")
        values = struct.unpack_from("<4s4H2LH", tail, eocd_pos)
        total_entries = values[4]
        central_size = values[5]
        central_offset = values[6]

    central = ranged_get(url, central_offset, central_offset + central_size - 1)
    entries: list[dict] = []
    offset = 0
    for _ in range(total_entries):
        if central[offset : offset + 4] != b"PK\x01\x02":
            raise RuntimeError(f"Invalid central directory entry at {offset}")
        values = struct.unpack_from("<4s2H4H3L5H2L", central, offset)
        method = values[4]
        compressed_size = values[8]
        uncompressed_size = values[9]
        name_len = values[10]
        extra_len = values[11]
        comment_len = values[12]
        local_header_offset = values[16]
        name_start = offset + 46
        name = central[name_start : name_start + name_len].decode("utf-8", "replace")
        extra = central[name_start + name_len : name_start + name_len + extra_len]

        if 0xFFFFFFFF in {compressed_size, uncompressed_size, local_header_offset}:
            cursor = 0
            while cursor + 4 <= len(extra):
                header_id, data_size = struct.unpack_from("<HH", extra, cursor)
                cursor += 4
                data = extra[cursor : cursor + data_size]
                cursor += data_size
                if header_id != 0x0001:
                    continue
                value_cursor = 0
                if uncompressed_size == 0xFFFFFFFF:
                    uncompressed_size = struct.unpack_from("<Q", data, value_cursor)[0]
                    value_cursor += 8
                if compressed_size == 0xFFFFFFFF:
                    compressed_size = struct.unpack_from("<Q", data, value_cursor)[0]
                    value_cursor += 8
                if local_header_offset == 0xFFFFFFFF:
                    local_header_offset = struct.unpack_from("<Q", data, value_cursor)[0]

        entries.append(
            {
                "name": name,
                "method": method,
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "local_header_offset": local_header_offset,
            }
        )
        offset += 46 + name_len + extra_len + comment_len
    return entries


def download_zip_entry(url: str, entry: dict) -> bytes:
    header = ranged_get(url, entry["local_header_offset"], entry["local_header_offset"] + 200)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"Invalid local ZIP header for {entry['name']}")
    values = struct.unpack_from("<4s5H3L2H", header, 0)
    method = values[3]
    name_len = values[9]
    extra_len = values[10]
    data_start = entry["local_header_offset"] + 30 + name_len + extra_len
    data = ranged_get(url, data_start, data_start + entry["compressed_size"] - 1)
    if method == 0:
        return data
    if method == 8:
        return zlib.decompress(data, -zlib.MAX_WBITS)
    raise RuntimeError(f"Unsupported ZIP compression method for {entry['name']}: {method}")


def zip_entry_data_bounds(url: str, entry: dict) -> tuple[int, int]:
    header = ranged_get(url, entry["local_header_offset"], entry["local_header_offset"] + 200)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"Invalid local ZIP header for {entry['name']}")
    values = struct.unpack_from("<4s5H3L2H", header, 0)
    name_len = values[9]
    extra_len = values[10]
    data_start = entry["local_header_offset"] + 30 + name_len + extra_len
    return data_start, data_start + entry["compressed_size"] - 1


def extract_zip_entry_from_span(span: bytes, span_start: int, entry: dict) -> bytes:
    header_offset = entry["local_header_offset"] - span_start
    header = span[header_offset : header_offset + 200]
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"Invalid local ZIP header for {entry['name']}")
    values = struct.unpack_from("<4s5H3L2H", header, 0)
    method = values[3]
    name_len = values[9]
    extra_len = values[10]
    data_start = header_offset + 30 + name_len + extra_len
    data = span[data_start : data_start + entry["compressed_size"]]
    if method == 0:
        return data
    if method == 8:
        return zlib.decompress(data, -zlib.MAX_WBITS)
    raise RuntimeError(f"Unsupported ZIP compression method for {entry['name']}: {method}")


def safe_extract_images_from_zip(zip_bytes: bytes, output_dir: Path, dataset: str, limit: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for info in archive.infolist():
            suffix = Path(info.filename).suffix.lower()
            if suffix not in IMAGE_SUFFIXES:
                continue
            target = output_dir / f"{dataset}_{Path(info.filename).name}"
            with archive.open(info) as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            paths.append(target)
            if len(paths) >= limit:
                break
    return paths


def download_idpl_samples(target_root: Path, repo_root: Path, per_source_limit: int) -> list[SourceImage]:
    items = request_with_retries("GET", IDPL_IMAGES_API).json()
    output_dir = target_root / "sources" / "idpl_pfod"
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[SourceImage] = []
    for item in items[:per_source_limit]:
        if item.get("type") != "file" or not item.get("download_url"):
            continue
        target = output_dir / item["name"]
        response = request_with_retries("GET", item["download_url"], stream=True)
        with target.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
        records.append(
            SourceImage(
                source_dataset="IDPL-PFOD",
                source_id=item["name"],
                language="Persian",
                image_path=repo_relative(target, repo_root),
                original_url=item["html_url"],
            )
        )
    return records


def download_openiti_subset(target_root: Path, repo_root: Path, max_bytes: int, per_source_limit: int) -> list[SourceImage]:
    record = request_with_retries("GET", OPENITI_RECORD_API).json()
    files = {item["key"]: item for item in record["files"]}
    metadata_url = files["OpenITI-Makhzan_Metadata_2026-1-2.tsv"]["links"]["self"]
    data_url = files["OpenITI-Makhzan_Data_2026-1-2.zip"]["links"]["self"]

    output_dir = target_root / "sources" / "openiti_makhzan"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "OpenITI-Makhzan_Metadata_2026-1-2.tsv"
    metadata_path.write_bytes(request_with_retries("GET", metadata_url).content)

    entries = parse_zip_central_directory(data_url)
    selected = [
        entry
        for entry in sorted(entries, key=lambda row: row["compressed_size"])
        if entry["name"].lower().endswith(".zip") and entry["compressed_size"] <= max_bytes
    ]

    records: list[SourceImage] = []
    spent = 0
    for entry in selected:
        if len(records) >= per_source_limit or spent + entry["compressed_size"] > max_bytes:
            break
        inner_zip = download_zip_entry(data_url, entry)
        spent += len(inner_zip)
        extracted = safe_extract_images_from_zip(inner_zip, output_dir, Path(entry["name"]).stem, limit=3)
        for image in extracted:
            records.append(
                SourceImage(
                    source_dataset="OpenITI-Makhzan",
                    source_id=f"{entry['name']}:{image.name}",
                    language="Arabic-script",
                    image_path=repo_relative(image, repo_root),
                    original_url=OPENITI_RECORD_API,
                )
            )
            if len(records) >= per_source_limit:
                break
    return records


def download_kaggle_subset(target_root: Path, repo_root: Path, max_bytes: int, per_source_limit: int) -> list[SourceImage]:
    redirect = requests.get(KAGGLE_DOWNLOAD_API, stream=True, allow_redirects=False, timeout=30)
    redirect.raise_for_status()
    archive_url = redirect.headers["location"]
    entries = parse_zip_central_directory(archive_url)
    image_entries = [
        entry
        for entry in entries
        if Path(entry["name"]).suffix.lower() in IMAGE_SUFFIXES and entry["compressed_size"] <= max_bytes
    ]

    output_dir = target_root / "sources" / "arabic_documents_ocr"
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[SourceImage] = []
    spent = 0
    for entry in image_entries:
        if len(records) >= per_source_limit or spent + entry["compressed_size"] > max_bytes:
            break
        data = download_zip_entry(archive_url, entry)
        spent += len(data)
        target = output_dir / Path(entry["name"]).name
        target.write_bytes(data)
        records.append(
            SourceImage(
                source_dataset="Arabic-Documents-OCR",
                source_id=entry["name"],
                language="Arabic",
                image_path=repo_relative(target, repo_root),
                original_url="https://www.kaggle.com/datasets/humansintheloop/arabic-documents-ocr-dataset",
            )
        )
    return records


def download_invoice_table_subset(target_root: Path, repo_root: Path, max_bytes: int, target_count: int) -> list[SourceImage]:
    redirect = requests.get(INVOICE_DOWNLOAD_API, stream=True, allow_redirects=False, timeout=30)
    redirect.raise_for_status()
    archive_url = redirect.headers["location"]
    entries = parse_zip_central_directory(archive_url)
    image_entries = [
        entry
        for entry in entries
        if Path(entry["name"]).suffix.lower() in IMAGE_SUFFIXES and entry["compressed_size"] <= max_bytes
    ]

    output_dir = target_root / "sources" / "invoice_tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    pending: list[dict] = []
    spent = 0
    for entry in image_entries:
        if len(existing) + len(pending) >= target_count:
            break
        if spent + entry["compressed_size"] > max_bytes:
            break
        target_name = f"{Path(entry['name']).parent.as_posix().replace('/', '_')}_{Path(entry['name']).name}"
        if target_name in existing:
            continue
        entry = dict(entry)
        entry["target_name"] = target_name
        pending.append(entry)
        spent += entry["compressed_size"]

    records: list[SourceImage] = []
    pending.sort(key=lambda row: row["local_header_offset"])
    max_batch_bytes = 64 * 1024 * 1024
    index = 0
    while index < len(pending):
        batch_start = pending[index]["local_header_offset"]
        end_index = index
        while end_index + 1 < len(pending):
            proposed_end = pending[end_index + 1]["local_header_offset"] - 1
            if proposed_end - batch_start + 1 > max_batch_bytes:
                break
            end_index += 1

        if end_index + 1 < len(pending):
            batch_end = pending[end_index + 1]["local_header_offset"] - 1
        else:
            _, batch_end = zip_entry_data_bounds(archive_url, pending[end_index])

        batch = pending[index : end_index + 1]
        span = ranged_get(archive_url, batch_start, batch_end)
        for item in batch:
            data = extract_zip_entry_from_span(span, batch_start, item)
            target = output_dir / item["target_name"]
            target.write_bytes(data)
            records.append(
                SourceImage(
                    source_dataset="Invoice-Tables-OCR",
                    source_id=item["name"],
                    language="Printed",
                    image_path=repo_relative(target, repo_root),
                    original_url="https://www.kaggle.com/datasets/osamahosamabdellatif/high-quality-invoice-images-for-ocr",
                )
            )
        index = end_index + 1
    return records


def build_rotated_eval(
    target_root: Path,
    repo_root: Path,
    source_records: Iterable[SourceImage],
    rotation_format: str,
    jpeg_quality: int,
    max_rotated_side: int,
) -> list[RotatedImage]:
    if rotation_format not in {"jpg", "png"}:
        raise ValueError("rotation_format must be 'jpg' or 'png'")
    rotated_dir = target_root / "rotated"
    if rotated_dir.exists():
        shutil.rmtree(rotated_dir)
    rotated_dir.mkdir(parents=True, exist_ok=True)
    labels: list[RotatedImage] = []
    for record in source_records:
        source_path = repo_root / record.image_path
        with Image.open(source_path) as image:
            rgb = image.convert("RGB")
            if max_rotated_side > 0:
                rgb.thumbnail((max_rotated_side, max_rotated_side), Image.Resampling.LANCZOS)
            for angle in (0, 90, 180, 270):
                output_name = f"{record.source_dataset.lower().replace('-', '_')}_{Path(record.image_path).stem}_rot{angle:03d}.{rotation_format}"
                output_path = rotated_dir / output_name
                rotated = rgb.rotate(angle, expand=True)
                if rotation_format == "jpg":
                    rotated.save(output_path, format="JPEG", quality=jpeg_quality, optimize=True)
                else:
                    rotated.save(output_path, format="PNG")
                labels.append(
                    RotatedImage(
                        image_path=repo_relative(output_path, repo_root),
                        true_angle=angle,
                        source_dataset=record.source_dataset,
                        source_id=record.source_id,
                        source_image_path=record.image_path,
                        language=record.language,
                    )
                )
    return labels


def collect_existing_sources(target_root: Path, repo_root: Path) -> list[SourceImage]:
    dataset_info = {
        "idpl_pfod": ("IDPL-PFOD", "Persian", "https://github.com/FtmsdtHosseini/IDPL-PFOD"),
        "openiti_makhzan": ("OpenITI-Makhzan", "Arabic-script", OPENITI_RECORD_API),
        "arabic_documents_ocr": (
            "Arabic-Documents-OCR",
            "Arabic",
            "https://www.kaggle.com/datasets/humansintheloop/arabic-documents-ocr-dataset",
        ),
        "invoice_tables": (
            "Invoice-Tables-OCR",
            "Printed",
            "https://www.kaggle.com/datasets/osamahosamabdellatif/high-quality-invoice-images-for-ocr",
        ),
    }
    records: list[SourceImage] = []
    for folder_name, (dataset_name, language, url) in dataset_info.items():
        folder = target_root / "sources" / folder_name
        if not folder.exists():
            continue
        for image_path in sorted(folder.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                records.append(
                    SourceImage(
                        source_dataset=dataset_name,
                        source_id=image_path.name,
                        language=language,
                        image_path=repo_relative(image_path, repo_root),
                        original_url=url,
                    )
                )
    return records


def write_manifests(target_root: Path, source_records: list[SourceImage], rotated_records: list[RotatedImage]) -> None:
    with (target_root / "sources_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in source_records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    with (target_root / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "true_angle", "source_dataset", "source_id", "source_image_path", "language"],
        )
        writer.writeheader()
        for record in rotated_records:
            writer.writerow(asdict(record))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a bounded Arabic/Persian orientation evaluation subset.")
    parser.add_argument("--output", type=Path, default=Path("data/raw/orientation_eval"))
    parser.add_argument("--max-bytes", type=int, default=GIB)
    parser.add_argument("--per-source-limit", type=int, default=40)
    parser.add_argument("--table-target", type=int, default=0, help="Target number of invoice/table source images to keep locally.")
    parser.add_argument("--tables-only", action="store_true", help="Only download/extend the invoice/table source.")
    parser.add_argument("--rotation-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=82)
    parser.add_argument("--max-rotated-side", type=int, default=1600)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--from-existing-only", action="store_true", help="Rebuild labels/manifests from already downloaded files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    if args.from_existing_only:
        source_records = collect_existing_sources(args.output, args.repo_root)
        print(f"Existing files: collected {len(source_records)} source image(s)")
    else:
        per_source_budget = args.max_bytes // 3
        downloaders = []
        if not args.tables_only:
            downloaders.extend(
                [
                    ("IDPL-PFOD", lambda: download_idpl_samples(args.output, args.repo_root, args.per_source_limit)),
                    ("OpenITI-Makhzan", lambda: download_openiti_subset(args.output, args.repo_root, per_source_budget, args.per_source_limit)),
                    ("Arabic-Documents-OCR", lambda: download_kaggle_subset(args.output, args.repo_root, per_source_budget, args.per_source_limit)),
                ]
            )
        downloaders.append(
            (
                "Invoice-Tables-OCR",
                lambda: download_invoice_table_subset(args.output, args.repo_root, args.max_bytes, args.table_target)
                if args.table_target > 0
                else [],
            )
        )
        for name, downloader in downloaders:
            try:
                records = downloader()
                print(f"{name}: downloaded {len(records)} source image(s)")
            except Exception as exc:
                failures.append(f"{name}: {exc}")
                print(f"{name}: skipped ({exc})")
        source_records = collect_existing_sources(args.output, args.repo_root)
        print(f"Existing files: collected {len(source_records)} source image(s)")

    rotated_records = build_rotated_eval(
        args.output,
        args.repo_root,
        source_records,
        args.rotation_format,
        args.jpeg_quality,
        args.max_rotated_side,
    )
    write_manifests(args.output, source_records, rotated_records)

    summary = {
        "source_images": len(source_records),
        "rotated_images": len(rotated_records),
        "failures": failures,
    }
    (args.output / "download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
