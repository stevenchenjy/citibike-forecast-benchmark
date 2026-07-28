"""Acquire the official research source without making it a runtime dependency."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from citibike_benchmark.constants import PROJECT_ROOT, SOURCE_RELATIVE_PATH, SOURCE_REPOSITORY_URL
from citibike_benchmark.utils.io import write_json


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    commit: str
    downloaded_at: str
    cloned: bool


def _git(args: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def download_source(
    destination: Path | None = None, repository_url: str = SOURCE_REPOSITORY_URL
) -> DownloadResult:
    """Shallow-clone the required source once, then report its exact commit.

    An existing checkout is deliberately not pulled automatically: changing the
    source revision would change benchmark inputs and must be an explicit action.
    """
    path = destination or PROJECT_ROOT / SOURCE_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    cloned = False
    if not path.exists():
        _git(["clone", "--depth", "1", repository_url, str(path)])
        cloned = True
    if not (path / ".git").exists():
        raise RuntimeError(f"Source path exists but is not a Git checkout: {path}")
    commit = _git(["rev-parse", "HEAD"], cwd=path)
    download = DownloadResult(
        path=path,
        commit=commit,
        downloaded_at=datetime.now(UTC).isoformat(),
        cloned=cloned,
    )
    write_json(PROJECT_ROOT / "data/manifests/source_download.json", {
        "repository_url": repository_url,
        "local_path": str(path.relative_to(PROJECT_ROOT)),
        "source_commit": download.commit,
        "download_date": download.downloaded_at,
        "cloned_this_invocation": download.cloned,
    })
    return download
