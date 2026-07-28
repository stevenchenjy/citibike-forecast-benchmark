"""Typer command-line interface for all reproducible project stages."""
from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from citibike_benchmark.config import load_config
from citibike_benchmark.data.download import download_source
from citibike_benchmark.data.panel import build_hourly_panel
from citibike_benchmark.data.source_adapter import inspect_source, write_data_audit
from citibike_benchmark.evaluation.backtest import run_backtest
from citibike_benchmark.evaluation.decision import run_decision_evaluation
from citibike_benchmark.reporting.report import build_report

app = typer.Typer(help="Reproducible Citi Bike forecasting and inventory benchmark.", no_args_is_help=True)


@app.command()
def download() -> None:
    """Shallow-clone the required official VP-RNN source repository."""
    result = download_source()
    state = "cloned" if result.cloned else "reused pinned checkout"
    print(f"{state}: {result.path} at {result.commit}")


@app.command()
def inspect() -> None:
    """Inspect the acquired source schema and create its data audit."""
    manifest = inspect_source()
    report = write_data_audit(manifest)
    print(f"Audited {len(manifest['station_ids'])} stations at {manifest['source_commit']}")
    print(f"Wrote {report}")


@app.command()
def build(config: Path = typer.Option(Path("configs/core.yaml"))) -> None:
    """Build the canonical processed hourly panel."""
    metadata = build_hourly_panel(config)
    cache = "reused cached" if metadata["cached"] else "built"
    print(f"{cache} {metadata['rows']:,}-row panel at {metadata['panel_path']}")


@app.command()
def backtest(config: Path = typer.Option(Path("configs/core.yaml"))) -> None:
    """Run chronological forecast backtests."""
    result = run_backtest(config)
    cache = "reused cached" if result["cached"] else "completed"
    print(f"{cache} forecast backtest {result['run_id']}: {result['prediction_path']}")


@app.command()
def decision(config: Path = typer.Option(Path("configs/core.yaml"))) -> None:
    """Run station-level inventory-decision evaluation."""
    result = run_decision_evaluation(config)
    print(f"Completed {result['station_days']} station-day decisions: {result['decision_metrics']}")


@app.command()
def report(config: Path = typer.Option(Path("configs/core.yaml"))) -> None:
    """Write final tables, figures, and report."""
    result = build_report(config)
    print(f"Wrote {result['report']}")


if __name__ == "__main__":
    app()
