#!/usr/bin/env python3
"""Run inventory evaluation through the benchmark CLI."""
from citibike_benchmark.cli import app

if __name__ == "__main__":
    app(["decision"])
