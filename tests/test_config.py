import pytest

from aura.config import (DEFAULT_SETTINGS, ConfigError, deep_merge, get_secret, load_safety, load_settings,
                         resolve_path)


def test_defaults_load_without_any_file(tmp_path, monkeypatch):
    s = load_settings()
    assert s["performance_mode"] == "BALANCED" and s["ai"]["enabled"] is False
    assert s["memory"]["enabled"] is True


def test_repository_config_files_are_valid():
    s, saf = load_settings(), load_safety()
    assert s["audio"]["stt_backend"] == "keyboard"
    assert saf["cooldown_seconds"]["CRITICAL"] < saf["cooldown_seconds"]["LOW"]


def test_yaml_overrides_defaults_and_keeps_siblings(tmp_path):
    f = tmp_path / "s.yaml"
    f.write_text("camera:\n  source: 3\nmemory:\n  retention_days: 7\n")
    s = load_settings(str(f))
    assert s["camera"]["source"] == 3 and s["camera"]["height"] == 480
    assert s["memory"]["retention_days"] == 7 and s["memory"]["enabled"] is True


@pytest.mark.parametrize("mode,size,skip", [("LOW_POWER", 320, 6), ("BALANCED", 416, 3), ("HIGH_PERFORMANCE", 640, 1)])
def test_performance_modes(mode, size, skip):
    s = load_settings(mode_override=mode)
    assert s["vision"]["inference_size"] == size and s["vision"]["process_every_n_frames"] == skip


def test_explicit_user_value_beats_mode_preset(tmp_path):
    f = tmp_path / "s.yaml"
    f.write_text("performance_mode: LOW_POWER\nvision:\n  inference_size: 512\n")
    s = load_settings(str(f))
    assert s["vision"]["inference_size"] == 512 and s["vision"]["process_every_n_frames"] == 6


def test_invalid_mode_rejected(tmp_path):
    f = tmp_path / "s.yaml"
    f.write_text("performance_mode: TURBO\n")
    with pytest.raises(ConfigError):
        load_settings(str(f))


def test_bad_yaml_and_missing_explicit_file(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [unclosed\n")
    with pytest.raises(ConfigError):
        load_settings(str(bad))
    with pytest.raises(ConfigError):
        load_settings(str(tmp_path / "missing.yaml"))
    lst = tmp_path / "list.yaml"
    lst.write_text("- 1\n- 2\n")
    with pytest.raises(ConfigError):
        load_safety(str(lst))


def test_safety_override(tmp_path):
    f = tmp_path / "saf.yaml"
    f.write_text("class_weights:\n  chair: 0.9\nmin_spoken_level: HIGH\n")
    saf = load_safety(str(f))
    assert saf["class_weights"]["chair"] == 0.9 and saf["class_weights"]["car"] == 1.0
    assert saf["min_spoken_level"] == "HIGH"


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1}}
    out = deep_merge(base, {"a": {"c": 2}})
    assert base == {"a": {"b": 1}} and out == {"a": {"b": 1, "c": 2}}


def test_secrets_come_only_from_environment(monkeypatch):
    monkeypatch.setenv("AURA_TEST_KEY", "  abc  ")
    assert get_secret("AURA_TEST_KEY") == "abc"
    monkeypatch.delenv("AURA_TEST_KEY")
    assert get_secret("AURA_TEST_KEY") is None
    assert "api_key" not in DEFAULT_SETTINGS["ai"]  # only the env var NAME is configurable


def test_resolve_path(tmp_path):
    assert resolve_path(str(tmp_path)) == tmp_path
    assert resolve_path("models/x.onnx").is_absolute()
