from core.config import load_config


def test_load_config_smoke() -> None:
    cfg = load_config()
    assert cfg["llm"]["model"] == "qwen3:8b"
    assert cfg["stt"]["port"] == 8771
