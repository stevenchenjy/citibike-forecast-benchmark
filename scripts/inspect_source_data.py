#!/usr/bin/env python3
"""Inspect source data through the benchmark CLI."""
import sys

from citibike_benchmark.cli import app

if __name__ == "__main__":
    sys.argv.insert(1, "inspect")
    app()
