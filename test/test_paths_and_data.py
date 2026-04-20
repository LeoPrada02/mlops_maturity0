from src.common.config import load_params, resolve_path


def test_params_load():
    params = load_params()
    assert "dataset" in params
    assert "model" in params
    assert "monitoring" in params


def test_raw_data_exists():
    params = load_params()
    raw_path = resolve_path(params["dataset"]["raw_path"])
    assert raw_path.exists()
