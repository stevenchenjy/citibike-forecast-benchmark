#!/usr/bin/env python3
"""Build reports through the benchmark CLI."""
from citibike_benchmark.cli import app

if __name__ == "__main__":
    app(["report"])
