"""Central constants for reproducible benchmark behavior."""
import os
from pathlib import Path


def _project_root() -> Path:
    """Resolve the checkout both for source and non-editable installations."""
    configured = os.environ.get("CITIBIKE_BENCHMARK_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
            return candidate
    # Fallback for editable/development contexts; normal project commands are
    # run from the checkout and therefore resolve through the branch above.
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _project_root()
SOURCE_REPOSITORY_URL = "https://github.com/DanieleGammelli/variational-poisson-rnn.git"
SOURCE_RELATIVE_PATH = Path("data/external/variational-poisson-rnn")
TIMEZONE = "America/New_York"
CANONICAL_COLUMNS = (
    "station_id", "timestamp", "date", "hour", "day_of_week", "is_weekend",
    "pickups", "returns", "net_flow", "station_capacity", "data_complete",
)
TARGETS = ("pickups", "returns")
REQUIRED_MODELS = (
    "seasonal_naive", "historical_average", "recent_average", "poisson_glm", "lightgbm_poisson",
)
WEATHER_EXPERIMENT_NAME = "observed_weather_hindsight_upper_bound"
