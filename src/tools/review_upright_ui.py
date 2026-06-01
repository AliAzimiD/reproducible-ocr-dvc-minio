from __future__ import annotations

import argparse
import csv
import base64
import json
import mimetypes
import sys
import threading
import time
import webbrowser
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
ROTATION_BY_ACTION = {
    "save": 0,
    "rotate_left": 270,
    "rotate_right": 90,
    "rotate_180": 180,
}


@dataclass(frozen=True)
class ReviewRecord:
    source_path: str
    output_path: str
    action: str
    rotation_degrees: int
    reviewed_at: str
    predicted_angle: int | None = None
    prediction_confidence: float | None = None


@dataclass(frozen=True)
class PredictionRecord:
    source_path: str
    predicted_angle: int | None
    confidence: float | None
    predicted_at: str


def repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def discover_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    images = [
        path
        for path in sorted(input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not images:
        raise ValueError(f"No images found under {input_dir}")
    return images


def rotate_and_save(source_path: Path, output_path: Path, rotation_degrees: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        image.convert("RGB").rotate(-rotation_degrees, expand=True).save(output_path)


def copy_or_save_upright(source_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        image.convert("RGB").save(output_path)


def image_to_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


def rotated_image_bytes(source_path: Path, rotation_degrees: int) -> tuple[bytes, str]:
    from io import BytesIO

    with Image.open(source_path) as image:
        output = BytesIO()
        image.convert("RGB").rotate(-rotation_degrees, expand=True).save(output, format="PNG")
        return output.getvalue(), "image/png"


class ReviewApp:
    def __init__(self, input_dir: Path, output_dir: Path, repo_root: Path, model_dir: Path | None = None) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.repo_root = repo_root
        self.images = discover_images(input_dir)
        self.predictions_path = output_dir / "predictions.csv"
        self.manifest_path = output_dir / "review_manifest.jsonl"
        self.state_path = output_dir / "review_state.json"
        self.lock = threading.RLock()
        self.index = 0
        self.preview_rotation = 0
        self.predictor = None
        if model_dir is not None:
            from src.stages.orient_and_rotate import PaddleOrientationPredictor

            self.predictor = PaddleOrientationPredictor(model_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.predictions = self._load_or_create_predictions()
        self.images = self._ordered_images()
        self._load_state()
        self._advance_to_next_unreviewed()

    def _load_or_create_predictions(self) -> dict[str, PredictionRecord]:
        if self.predictor is None:
            return {
                repo_relative(path, self.repo_root): PredictionRecord(
                    source_path=repo_relative(path, self.repo_root),
                    predicted_angle=None,
                    confidence=None,
                    predicted_at=datetime.now(timezone.utc).isoformat(),
                )
                for path in self.images
            }

        existing = self._load_predictions_file()
        missing = [
            path
            for path in self.images
            if repo_relative(path, self.repo_root) not in existing
        ]
        if not missing:
            return existing

        predictions = dict(existing)
        for image_path in missing:
            predicted_angle, confidence = self._prediction(image_path)
            relative_path = repo_relative(image_path, self.repo_root)
            predictions[relative_path] = PredictionRecord(
                source_path=relative_path,
                predicted_angle=predicted_angle,
                confidence=confidence,
                predicted_at=datetime.now(timezone.utc).isoformat(),
            )
        self._write_predictions_file(predictions)
        return predictions

    def _load_predictions_file(self) -> dict[str, PredictionRecord]:
        if not self.predictions_path.exists():
            return {}
        records: dict[str, PredictionRecord] = {}
        with self.predictions_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                angle = row.get("predicted_angle")
                confidence = row.get("confidence")
                records[row["source_path"]] = PredictionRecord(
                    source_path=row["source_path"],
                    predicted_angle=None if angle in {"", None} else int(angle),
                    confidence=None if confidence in {"", None} else float(confidence),
                    predicted_at=row.get("predicted_at", ""),
                )
        return records

    def _write_predictions_file(self, predictions: dict[str, PredictionRecord]) -> None:
        self.predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with self.predictions_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["source_path", "predicted_angle", "confidence", "predicted_at"],
            )
            writer.writeheader()
            for record in sorted(predictions.values(), key=lambda item: item.source_path):
                writer.writerow(asdict(record))

    def _ordered_images(self) -> list[Path]:
        def sort_key(path: Path) -> tuple[int, float, str]:
            relative = repo_relative(path, self.repo_root)
            prediction = self.predictions.get(relative)
            angle = None if prediction is None else prediction.predicted_angle
            confidence = None if prediction is None else prediction.confidence
            non_zero_group = 0 if angle not in {None, 0} else 1
            confidence_value = 1.0 if confidence is None else confidence
            return non_zero_group, confidence_value, relative

        return sorted(self.images, key=sort_key)

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.index = int(state.get("index", 0))
        except (ValueError, json.JSONDecodeError):
            self.index = 0

    def _save_state(self) -> None:
        self.state_path.write_text(json.dumps({"index": self.index}, indent=2), encoding="utf-8")

    def output_path_for(self, source_path: Path) -> Path:
        try:
            relative = source_path.resolve().relative_to(self.input_dir.resolve())
        except ValueError:
            relative = Path(source_path.name)
        return self.output_dir / relative

    def reviewed_count(self) -> int:
        return sum(1 for image in self.images if self.output_path_for(image).exists())

    def current_image(self) -> Path | None:
        if self.index >= len(self.images):
            return None
        return self.images[self.index]

    def _advance_to_next_unreviewed(self) -> None:
        while self.index < len(self.images) and self.output_path_for(self.images[self.index]).exists():
            self.index += 1
        self.preview_rotation = 0
        self._save_state()

    def _prediction(self, image_path: Path) -> tuple[int | None, float | None]:
        if self.predictor is None:
            return None, None
        return self.predictor.predict(image_path)

    def _cached_prediction(self, image_path: Path) -> tuple[int | None, float | None]:
        record = self.predictions.get(repo_relative(image_path, self.repo_root))
        if record is None:
            return None, None
        return record.predicted_angle, record.confidence

    def status(self) -> dict[str, Any]:
        with self.lock:
            image_path = self.current_image()
            if image_path is None:
                return {
                    "done": True,
                    "reviewed": self.reviewed_count(),
                    "total": len(self.images),
                    "message": "All images reviewed.",
                }
            predicted_angle, confidence = self._cached_prediction(image_path)
            return {
                "done": False,
                "index": self.index + 1,
                "reviewed": self.reviewed_count(),
                "total": len(self.images),
                "source_path": repo_relative(image_path, self.repo_root),
                "output_path": repo_relative(self.output_path_for(image_path), self.repo_root),
                "image_url": "/image/current",
                "preview_rotation": self.preview_rotation,
                "predicted_angle": predicted_angle,
                "prediction_confidence": confidence,
            }

    def preview_current(self, action: str) -> dict[str, Any]:
        with self.lock:
            if action == "auto_fix":
                image_path = self.current_image()
                if image_path is None:
                    return self.status()
                predicted_angle, _ = self._cached_prediction(image_path)
                if predicted_angle is None:
                    raise ValueError("Auto Fix requires --model-dir.")
                self.preview_rotation = predicted_angle % 360
            elif action == "rotate_left":
                self.preview_rotation = (self.preview_rotation + 270) % 360
            elif action == "rotate_right":
                self.preview_rotation = (self.preview_rotation + 90) % 360
            elif action == "rotate_180":
                self.preview_rotation = (self.preview_rotation + 180) % 360
            elif action == "reset":
                self.preview_rotation = 0
            else:
                raise ValueError(f"Unsupported preview action: {action}")
            return self.status()

    def review_current(self, action: str) -> dict[str, Any]:
        with self.lock:
            image_path = self.current_image()
            if image_path is None:
                return self.status()
            predicted_angle, confidence = self._cached_prediction(image_path)
            if action == "save_preview":
                rotation_degrees = self.preview_rotation
                manifest_action = "save_preview"
            else:
                if action not in ROTATION_BY_ACTION:
                    raise ValueError(f"Unsupported action: {action}")
                rotation_degrees = ROTATION_BY_ACTION[action]
                manifest_action = action

            output_path = self.output_path_for(image_path)
            if rotation_degrees == 0:
                copy_or_save_upright(image_path, output_path)
            else:
                rotate_and_save(image_path, output_path, rotation_degrees)

            record = ReviewRecord(
                source_path=repo_relative(image_path, self.repo_root),
                output_path=repo_relative(output_path, self.repo_root),
                action=manifest_action,
                rotation_degrees=rotation_degrees,
                reviewed_at=datetime.now(timezone.utc).isoformat(),
                predicted_angle=predicted_angle,
                prediction_confidence=confidence,
            )
            with self.manifest_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")

            self.index += 1
            self._advance_to_next_unreviewed()
            return self.status()

    def previous(self) -> dict[str, Any]:
        with self.lock:
            self.index = max(0, self.index - 1)
            self.preview_rotation = 0
            self._save_state()
            return self.status()

    def current_image_bytes(self) -> tuple[bytes, str]:
        image_path = self.current_image()
        if image_path is None:
            raise FileNotFoundError("No current image")
        if self.preview_rotation:
            return rotated_image_bytes(image_path, self.preview_rotation)
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        return image_path.read_bytes(), mime_type


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Upright Image Review</title>
  <style>
    :root { color-scheme: light; font-family: Inter, Segoe UI, Arial, sans-serif; }
    body { margin: 0; background: #f6f7f9; color: #111827; }
    header { padding: 18px 24px; border-bottom: 1px solid #d9dde5; background: #ffffff; display: flex; justify-content: space-between; gap: 16px; align-items: center; }
    h1 { font-size: 20px; margin: 0; font-weight: 650; letter-spacing: 0; }
    main { display: grid; grid-template-columns: minmax(0, 1fr) 340px; min-height: calc(100vh - 70px); }
    .viewer { display: flex; align-items: center; justify-content: center; padding: 24px; overflow: auto; }
    .viewer img { max-width: 100%; max-height: calc(100vh - 130px); box-shadow: 0 8px 24px rgba(15, 23, 42, 0.14); background: white; }
    aside { border-left: 1px solid #d9dde5; background: #ffffff; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
    .meta { display: grid; gap: 8px; font-size: 13px; color: #4b5563; word-break: break-word; }
    .progress { font-size: 14px; font-weight: 600; color: #111827; }
    .buttons { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    button { border: 1px solid #b9c0cc; background: #ffffff; color: #111827; padding: 11px 12px; border-radius: 6px; font-size: 14px; cursor: pointer; }
    button:hover { background: #f0f3f7; }
    button.primary { grid-column: span 2; background: #14532d; color: #ffffff; border-color: #14532d; }
    button.primary:hover { background: #166534; }
    button.fix { background: #1d4ed8; color: #ffffff; border-color: #1d4ed8; }
    button.fix:hover { background: #1e40af; }
    .done { padding: 32px; font-size: 18px; text-align: center; color: #374151; }
    .rotation { font-size: 14px; font-weight: 600; color: #1f2937; }
    code { font-size: 12px; }
  </style>
</head>
<body>
  <header>
    <h1>Upright Image Review</h1>
    <div id="progress" class="progress"></div>
  </header>
  <main>
    <section id="viewer" class="viewer"></section>
    <aside>
      <div class="meta">
        <div><strong>Source</strong><br><code id="source"></code></div>
        <div><strong>Output</strong><br><code id="output"></code></div>
        <div><strong>Model suggestion</strong><br><span id="prediction"></span></div>
        <div><strong>Preview rotation</strong><br><span id="rotation" class="rotation"></span></div>
      </div>
      <div class="buttons">
        <button class="primary" onclick="saveReviewed()">Save Reviewed</button>
        <button class="fix" onclick="preview('auto_fix')" id="autoFix">Auto Fix</button>
        <button onclick="preview('rotate_left')">Rotate Left</button>
        <button onclick="preview('rotate_right')">Rotate Right</button>
        <button onclick="preview('rotate_180')">Rotate 180</button>
        <button onclick="preview('reset')">Reset</button>
        <button onclick="previous()">Back</button>
      </div>
    </aside>
  </main>
<script>
async function loadCurrent() {
  const response = await fetch('/api/current');
  const data = await response.json();
  document.getElementById('progress').textContent = `${data.reviewed} / ${data.total} reviewed`;
  if (data.done) {
    document.getElementById('viewer').innerHTML = `<div class="done">${data.message}</div>`;
    document.getElementById('source').textContent = '';
    document.getElementById('output').textContent = '';
    document.getElementById('prediction').textContent = '';
    document.getElementById('rotation').textContent = '';
    return;
  }
  document.getElementById('viewer').innerHTML = `<img src="${data.image_url}?t=${Date.now()}" alt="Current image">`;
  document.getElementById('source').textContent = data.source_path;
  document.getElementById('output').textContent = data.output_path;
  document.getElementById('rotation').textContent = `${data.preview_rotation || 0} degrees`;
  const autoFix = document.getElementById('autoFix');
  if (data.predicted_angle === null || data.predicted_angle === undefined) {
    document.getElementById('prediction').textContent = 'No model configured';
    autoFix.disabled = true;
  } else {
    const confidence = data.prediction_confidence === null ? '' : ` (${(data.prediction_confidence * 100).toFixed(1)}%)`;
    document.getElementById('prediction').textContent = `${data.predicted_angle} degrees${confidence}`;
    autoFix.disabled = false;
  }
}
async function preview(action) {
  const response = await fetch(`/api/preview?action=${encodeURIComponent(action)}`, { method: 'POST' });
  if (!response.ok) alert(await response.text());
  await loadCurrent();
}
async function saveReviewed() {
  const response = await fetch('/api/action?action=save_preview', { method: 'POST' });
  if (!response.ok) alert(await response.text());
  await loadCurrent();
}
async function previous() {
  const response = await fetch('/api/previous', { method: 'POST' });
  if (!response.ok) alert(await response.text());
  await loadCurrent();
}
loadCurrent();
</script>
</body>
</html>
"""


class ReviewRequestHandler(BaseHTTPRequestHandler):
    app: ReviewApp

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_bytes(self, data: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict[str, Any]) -> None:
        self._send_bytes(json.dumps(payload).encode("utf-8"), "application/json")

    def _send_error_text(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        self._send_bytes(message.encode("utf-8"), "text/plain; charset=utf-8", status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/api/current":
                self._send_json(self.app.status())
            elif parsed.path == "/image/current":
                data, mime_type = self.app.current_image_bytes()
                self._send_bytes(data, mime_type)
            else:
                self._send_error_text("Not found", HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_error_text(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/action":
                action = parse_qs(parsed.query).get("action", [""])[0]
                self._send_json(self.app.review_current(action))
            elif parsed.path == "/api/preview":
                action = parse_qs(parsed.query).get("action", [""])[0]
                self._send_json(self.app.preview_current(action))
            elif parsed.path == "/api/previous":
                self._send_json(self.app.previous())
            else:
                self._send_error_text("Not found", HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_error_text(str(exc), HTTPStatus.BAD_REQUEST)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a local UI for reviewing and correcting image orientation.")
    parser.add_argument("--input", type=Path, default=Path("data/interim/images_flat"), help="Input image folder.")
    parser.add_argument("--output", type=Path, default=Path("data/interim/images_flat_corrected_upright"), help="Output folder for reviewed upright images.")
    parser.add_argument("--model-dir", type=Path, default=None, help="Optional ONNX model directory for Auto Fix suggestions.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = ReviewApp(args.input, args.output, args.repo_root, args.model_dir)
    ReviewRequestHandler.app = app
    server = ThreadingHTTPServer((args.host, args.port), ReviewRequestHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Review UI running at {url}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping review UI.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
