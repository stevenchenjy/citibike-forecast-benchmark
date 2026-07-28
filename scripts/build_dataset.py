#!/usr/bin/env python3
"""Build a processed panel through the benchmark CLI."""
import sys

from citibike_benchmark.cli import app

if __name__ == "__main__":
    sys.argv.insert(1, "build")
    app()
