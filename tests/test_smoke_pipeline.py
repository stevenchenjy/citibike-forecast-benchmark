from citibike_benchmark.config import load_config


def test_smoke_profile_declares_required_no_weather_shape() -> None:
    config = load_config("configs/smoke.yaml")
    assert config["data"]["station_count"] == 5
    assert config["splits"]["folds"] == 1
    assert config["weather"]["enabled"] is False
