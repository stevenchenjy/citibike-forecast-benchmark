#!/usr/bin/env python3
"""Run forecast backtests through the benchmark CLI."""
import sys

from citibike_benchmark.cli import app

if __name__ == "__main__":
    sys.argv.insert(1, "backtest")
    app()
