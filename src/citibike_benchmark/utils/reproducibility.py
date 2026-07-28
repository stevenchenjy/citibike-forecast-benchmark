"""Run-manifest creation."""
from __future__ import annotations

import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
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


def enrich_experiment_manifest(run_id: str, config_path: Path, extra: dict[str, Any]) -> Path:
    """Write the complete audit manifest for an accepted saved experiment."""
    config_path = config_path.resolve()
    source_manifest_path = PROJECT_ROOT / "data/manifests/source_manifest.json"
    source_manifest = __import__("json").loads(source_manifest_path.read_text(encoding="utf-8"))
    package_names = ("numpy", "pandas", "polars", "scikit-learn", "lightgbm", "pyarrow", "pytest", "torch")
    package_versions: dict[str, str | None] = {}
    for package in package_names:
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = None
    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "local_repository_git_commit": _git("rev-parse", "HEAD"),
        "dirty_working_tree": bool(_git("status", "--porcelain")),
        "source_data_repository_commit": source_manifest["source_commit"],
        "source_data_repository_url": source_manifest["source_repository_url"],
        "config_path": str(config_path.relative_to(PROJECT_ROOT)),
        "config_contents": __import__("yaml").safe_load(config_path.read_text(encoding="utf-8")),
        "config_sha256": sha256_file(config_path),
        "package_versions": package_versions,
        "python": sys.version,
        "platform": platform.platform(),
        "hardware": {"logical_cpus": psutil.cpu_count(), "memory_bytes": psutil.virtual_memory().total},
        "input_file_hashes": source_manifest["input_file_sha256"],
        **extra,
    }
    output = PROJECT_ROOT / "artifacts/run_manifests" / f"{run_id}.json"
    write_json(output, manifest)
    return output
