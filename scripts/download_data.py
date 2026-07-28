#!/usr/bin/env python3
"""Download the pinned official source through the benchmark CLI."""
from citibike_benchmark.cli import app

if __name__ == "__main__":
    app(["download"])
