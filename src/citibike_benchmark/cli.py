"""Typer command-line interface for all reproducible project stages."""
from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from citibike_benchmark.config import load_config
from citibike_benchmark.data.download import download_source
from citibike_benchmark.data.source_adapter import inspect_source, write_data_audit

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
    load_config(config)
    raise typer.Exit("Canonical-panel construction is completed in Milestone 2.")


@app.command()
def backtest(config: Path = typer.Option(Path("configs/core.yaml"))) -> None:
    """Run chronological forecast backtests."""
    load_config(config)
    raise typer.Exit("Forecast backtesting is completed in Milestones 3 and 4.")


@app.command()
def decision(config: Path = typer.Option(Path("configs/core.yaml"))) -> None:
    """Run station-level inventory-decision evaluation."""
    load_config(config)
    raise typer.Exit("Decision evaluation is completed in Milestone 5.")


@app.command()
def report(config: Path = typer.Option(Path("configs/core.yaml"))) -> None:
    """Write final tables, figures, and report."""
    load_config(config)
    raise typer.Exit("Report generation is completed after evaluation milestones.")


if __name__ == "__main__":
    app()
