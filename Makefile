PYTHON ?= uv run --no-editable python
CLI = $(PYTHON) -m citibike_benchmark.cli

.PHONY: setup data inspect test smoke core gru decision report reproduce-smoke reproduce-core

setup:
	bash scripts/bootstrap.sh

data:
	$(CLI) download

inspect:
	$(CLI) inspect

test:
	uv run --no-editable --extra dev pytest

smoke:
	$(CLI) build --config configs/smoke.yaml
	$(CLI) backtest --config configs/smoke.yaml
	$(CLI) decision --config configs/smoke.yaml
	$(CLI) report --config configs/smoke.yaml

core:
	$(CLI) build --config configs/core.yaml
	$(CLI) backtest --config configs/core.yaml

gru:
	$(CLI) build --config configs/poisson_gru.yaml
	$(CLI) backtest --config configs/poisson_gru.yaml
	$(CLI) decision --config configs/poisson_gru.yaml
	$(CLI) report --config configs/core.yaml

decision:
	$(CLI) decision --config configs/core.yaml

report:
	$(CLI) report --config configs/core.yaml

reproduce-smoke:
	bash scripts/reproduce_smoke.sh

reproduce-core:
	bash scripts/reproduce_core.sh
