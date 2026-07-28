"""Central constants for reproducible benchmark behavior."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
