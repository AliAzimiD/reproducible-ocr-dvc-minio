from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration


def run(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, check=True)


def docker_daemon_available() -> bool:
    return subprocess.run(["docker", "info"], text=True, capture_output=True).returncode == 0


@pytest.mark.skipif(os.getenv("RUN_DVC_MINIO_TESTS") != "1", reason="set RUN_DVC_MINIO_TESTS=1 to run Docker/MinIO integration test")
def test_dvc_repro_and_push_to_minio(tmp_path: Path) -> None:
    if not docker_daemon_available():
        pytest.skip("Docker daemon is not available")

    repo_root = Path(__file__).resolve().parents[1]
    work = tmp_path / "repo"
    ignore = shutil.ignore_patterns(".git", ".dvc", ".venv", "data/interim", "__pycache__", ".pytest_cache")
    shutil.copytree(repo_root, work, ignore=ignore)

    run(["dvc", "init", "--no-scm", "-f"], work)
    run(["docker", "compose", "-f", "docker-compose.minio.yml", "up", "-d"], work)
    try:
        run(["docker", "compose", "-f", "docker-compose.minio.yml", "exec", "-T", "minio", "mc", "alias", "set", "local", "http://127.0.0.1:9000", "minioadmin", "minioadmin123"], work)
        run(["docker", "compose", "-f", "docker-compose.minio.yml", "exec", "-T", "minio", "mc", "mb", "--ignore-existing", "local/ocr-pipeline"], work)

        run(["dvc", "remote", "add", "-d", "-f", "minio", "s3://ocr-pipeline"], work)
        run(["dvc", "remote", "modify", "minio", "endpointurl", "http://127.0.0.1:9000"], work)
        run(["dvc", "remote", "modify", "minio", "access_key_id", "minioadmin"], work)
        run(["dvc", "remote", "modify", "minio", "secret_access_key", "minioadmin123"], work)
        run(["dvc", "repro"], work)
        run(["dvc", "push"], work)

        assert (work / "data" / "interim" / "images_flat" / "manifest.jsonl").exists()
        assert (work / "data" / "interim" / "orientation" / "manifest.jsonl").exists()
        assert (work / "data" / "interim" / "orientation" / "preds.csv").exists()
    finally:
        subprocess.run(["docker", "compose", "-f", "docker-compose.minio.yml", "down", "-v"], cwd=work, check=False)
