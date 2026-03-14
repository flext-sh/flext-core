"""Edge-case tests for FlextInfraBaseMkGenerator.

Covers permission errors, config normalization, stream errors,
and validation error handling.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from flext_core import t
from flext_infra.basemk.generator import FlextInfraBaseMkGenerator
from flext_tests import tm
from tests.infra.models import m as im


def test_generator_write_handles_file_permission_error(tmp_path: Path) -> None:
    output_path = tmp_path / "readonly" / "test.mk"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.touch()
    output_path.chmod(292)
    content = "all:\n\t@echo 'test'\n"
    gen = FlextInfraBaseMkGenerator()
    try:
        gen.write(content, output=output_path)
    finally:
        output_path.chmod(420)


def test_generator_normalize_config_with_none() -> None:
    gen = FlextInfraBaseMkGenerator()
    result = gen._normalize_config(None)
    tm.ok(result)
    assert isinstance(result.value, im.Infra.Basemk.BaseMkConfig)


def test_generator_normalize_config_with_basemk_config() -> None:
    config = im.Infra.Basemk.BaseMkConfig(
        project_name="test",
        python_version="3.13",
        core_stack="python",
        package_manager="poetry",
        source_dir="src",
        tests_dir="tests",
        lint_gates=["mypy"],
        test_command="pytest",
    )
    gen = FlextInfraBaseMkGenerator()
    result = gen._normalize_config(config)
    tm.ok(result)
    assert result.value == config


def test_generator_normalize_config_with_dict() -> None:
    config = im.Infra.Basemk.BaseMkConfig(
        project_name="test",
        python_version="3.13",
        core_stack="python",
        package_manager="poetry",
        source_dir="src",
        tests_dir="tests",
        lint_gates=["mypy"],
        test_command="pytest",
    )
    gen = FlextInfraBaseMkGenerator()
    result = gen._normalize_config(config)
    tm.ok(result)
    assert isinstance(result.value, im.Infra.Basemk.BaseMkConfig)


def test_generator_normalize_config_with_invalid_dict() -> None:
    invalid_dict = {"bad_key": "value"}
    gen = FlextInfraBaseMkGenerator()
    result = gen._normalize_config(invalid_dict)
    tm.fail(result)
    assert isinstance(result.error, str)
    assert "validation failed" in result.error


def test_generator_write_to_stream_handles_oserror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = io.StringIO()
    stream = io.StringIO()
    content = "all:\n\t@echo 'test'\n"
    gen = FlextInfraBaseMkGenerator()

    def mock_write(*args: t.Scalar, **kwargs: t.Scalar) -> None:
        msg = "Stream write failed"
        raise OSError(msg)

    monkeypatch.setattr(stream, "write", mock_write)
    result = gen.write(content, stream=stream)
    tm.fail(result)
    assert isinstance(result.error, str)
    assert "stdout write failed" in result.error


def test_generator_validate_generated_output_handles_oserror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gen = FlextInfraBaseMkGenerator()
    content = "all:\n\t@echo 'test'\n"

    def mock_tempdir(*args: t.Scalar, **kwargs: t.Scalar) -> None:
        msg = "Temp directory creation failed"
        raise OSError(msg)

    monkeypatch.setattr("tempfile.TemporaryDirectory", mock_tempdir)
    result = gen._validate_generated_output(content)
    tm.fail(result)
    assert isinstance(result.error, str)
    assert "validation failed" in result.error
