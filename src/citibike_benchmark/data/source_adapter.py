"""Source-layout discovery. Field mappings are set only after inspection."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

TABULAR_SUFFIXES = {".csv", ".tsv", ".txt", ".parquet", ".pkl", ".pickle", ".npy", ".npz"}


@dataclass(frozen=True)
class SourceFile:
    path: Path
    suffix: str
    bytes: int


def discover_source_files(source_root: Path) -> list[SourceFile]:
    """Return deterministic inventory of potentially relevant source files."""
    files: Iterable[Path] = source_root.rglob("*")
    return [
        SourceFile(path=path, suffix=path.suffix.lower(), bytes=path.stat().st_size)
        for path in sorted(files)
        if path.is_file() and path.suffix.lower() in TABULAR_SUFFIXES
    ]
