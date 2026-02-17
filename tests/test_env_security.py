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
