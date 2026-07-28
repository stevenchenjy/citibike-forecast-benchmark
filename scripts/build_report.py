#!/usr/bin/env python3
"""Build reports through the benchmark CLI."""
import sys

from citibike_benchmark.cli import app

if __name__ == "__main__":
    sys.argv.insert(1, "report")
    app()
