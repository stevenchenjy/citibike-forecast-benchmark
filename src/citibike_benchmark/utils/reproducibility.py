"""Run-manifest creation."""
from __future__ import annotations

import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from citibike_benchmark.constants import PROJECT_ROOT
from citibike_benchmark.utils.io import sha256_file, write_json


def _git(*args: str) -> str | None:
    result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def create_run_manifest(run_id: str, config_path: Path, extra: dict[str, Any] | None = None) -> Path:
    config_bytes_hash = sha256_file(config_path)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "repository_git_commit": _git("rev-parse", "HEAD"),
        "dirty_working_tree": bool(_git("status", "--porcelain")),
        "config_path": str(config_path),
        "config_sha256": config_bytes_hash,
        "python": sys.version,
        "platform": platform.platform(),
        "hardware": {"logical_cpus": psutil.cpu_count(), "memory_bytes": psutil.virtual_memory().total},
        "random_seeds": {},
        "warnings": [],
        "fallbacks": [],
    }
    if extra:
        manifest.update(extra)
    output = PROJECT_ROOT / "artifacts/run_manifests" / f"{run_id}.json"
    write_json(output, manifest)
    return output
