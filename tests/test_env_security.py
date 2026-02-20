import os
from pathlib import Path

import pytest

from mad_spark_alt.unified_cli import load_env_file


def test_secure_env_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure env is clean for test keys
    keys = ["SIMPLE_KEY", "WITH_INLINE_COMMENT", "QUOTED", "ESCAPED_QUOTES"]
    for key in keys:
        monkeypatch.delenv(key, raising=False)

    env_content = """
# This is a comment
SIMPLE_KEY=simple_value
WITH_INLINE_COMMENT=value # comment
QUOTED="quoted value"
ESCAPED_QUOTES="value with \\"quotes\\" inside"
"""
    env_file = tmp_path / ".env"
    env_file.write_text(env_content, encoding="utf-8")

    load_env_file(env_file)

    assert os.environ["SIMPLE_KEY"] == "simple_value"
    assert (
        os.environ["WITH_INLINE_COMMENT"] == "value"
    ), f"Expected 'value', got '{os.environ.get('WITH_INLINE_COMMENT')}'"
    assert (
        os.environ["QUOTED"] == "quoted value"
    ), f"Expected 'quoted value', got '{os.environ.get('QUOTED')}'"
    assert (
        os.environ["ESCAPED_QUOTES"] == 'value with "quotes" inside'
    ), f"Expected 'value with \"quotes\" inside', got '{os.environ.get('ESCAPED_QUOTES')}'"


def test_no_override_existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_content = "EXISTING_KEY=new_value"
    env_file = tmp_path / ".env"
    env_file.write_text(env_content, encoding="utf-8")

    monkeypatch.setenv("EXISTING_KEY", "original_value")

    load_env_file(env_file)

    assert os.environ["EXISTING_KEY"] == "original_value"


def test_directory_named_env_is_skipped(tmp_path: Path) -> None:
    """A directory named .env must not be passed to load_dotenv."""
    env_dir = tmp_path / ".env"
    env_dir.mkdir()

    # Should not raise — is_file() check skips directories
    load_env_file(env_dir)


def test_unreadable_env_file_logs_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """An unreadable .env file should log a warning instead of crashing."""
    from unittest.mock import patch
    from dotenv import load_dotenv as _load_dotenv

    env_file = tmp_path / ".env"
    env_file.write_text("KEY=value", encoding="utf-8")

    with patch("mad_spark_alt.unified_cli.load_dotenv", side_effect=OSError("Permission denied")):
        import logging
        with caplog.at_level(logging.WARNING, logger="mad_spark_alt.unified_cli"):
            load_env_file(env_file)

    assert any("Failed to load .env file" in r.message for r in caplog.records)
