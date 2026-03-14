"""Tests for FlextInfraBaseMkGenerator to achieve full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import io
from pathlib import Path

from flext_core import r
from flext_infra.basemk.generator import FlextInfraBaseMkGenerator
from flext_tests import tm
from tests.infra.models import m as im


class _SuccessRenderer:
    """Mock renderer that returns valid make syntax."""

    def render_all(self, config: im.Infra.Basemk.BaseMkConfig | None = None) -> r[str]:
        del config
        return r[str].ok(".PHONY: help\nhelp:\n\t@echo 'help'\n")


class _FailureRenderer:
    """Mock renderer that returns failure."""

    def render_all(self, config: im.Infra.Basemk.BaseMkConfig | None = None) -> r[str]:
        del config
        return r[str].fail("render error")


def test_generator_initializes_with_default_engine() -> None:
    """Test generator initializes with default TemplateEngine."""
    gen = FlextInfraBaseMkGenerator()
    assert gen is not None


def test_generator_initializes_with_custom_engine() -> None:
    """Test generator initializes with custom template engine."""
    mock_engine = _SuccessRenderer()
    gen = FlextInfraBaseMkGenerator(template_engine=mock_engine)
    assert gen is not None


def test_generator_execute_returns_generated_content() -> None:
    """Test execute() method returns generated content."""
    gen = FlextInfraBaseMkGenerator(template_engine=_SuccessRenderer())
    result = gen.execute()
    tm.ok(result)
    assert "help:" in result.value


def test_generator_generate_with_none_config_uses_default() -> None:
    """Test generate() with None config uses default configuration."""
    gen = FlextInfraBaseMkGenerator(template_engine=_SuccessRenderer())
    result = gen.generate(config=None)
    tm.ok(result)


def test_generator_generate_with_basemk_config_object() -> None:
    """Test generate() accepts BaseMkConfig object directly."""
    config = im.Infra.Basemk.BaseMkConfig(
        project_name="test-proj",
        python_version="3.13",
        core_stack="python",
        package_manager="poetry",
        source_dir="src",
        tests_dir="tests",
        lint_gates=["mypy"],
        test_command="pytest",
    )
    gen = FlextInfraBaseMkGenerator(template_engine=_SuccessRenderer())
    result = gen.generate(config=config)
    tm.ok(result)


def test_generator_generate_with_dict_config() -> None:
    """Test generate() accepts dict configuration."""
    config = im.Infra.Basemk.BaseMkConfig(
        project_name="dict-proj",
        python_version="3.13",
        core_stack="python",
        package_manager="poetry",
        source_dir="src",
        tests_dir="tests",
        lint_gates=["mypy"],
        test_command="pytest",
    )
    gen = FlextInfraBaseMkGenerator(template_engine=_SuccessRenderer())
    result = gen.generate(config=config)
    tm.ok(result)


def test_generator_generate_with_invalid_dict_config() -> None:
    """Test generate() fails with invalid dict configuration."""
    invalid_config = {"invalid_key": "value"}
    gen = FlextInfraBaseMkGenerator(template_engine=_SuccessRenderer())
    result = gen.generate(config=invalid_config)
    tm.fail(result)
    assert isinstance(result.error, str)
    assert isinstance(result.error, str)
    assert "validation failed" in result.error


def test_generator_generate_propagates_render_failure() -> None:
    """Test generate() propagates template render failures."""
    gen = FlextInfraBaseMkGenerator(template_engine=_FailureRenderer())
    result = gen.generate()
    tm.fail(result)
    assert isinstance(result.error, str)
    assert "render error" in result.error


def test_generator_write_to_file(tmp_path: Path) -> None:
    """Test write() saves content to file."""
    output_path = tmp_path / "test.mk"
    content = "all:\n\t@echo 'test'\n"
    gen = FlextInfraBaseMkGenerator()
    result = gen.write(content, output=output_path)
    tm.ok(result)
    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == content


def test_generator_write_creates_parent_directories(tmp_path: Path) -> None:
    """Test write() creates parent directories if needed."""
    output_path = tmp_path / "nested" / "dir" / "test.mk"
    content = "all:\n\t@echo 'test'\n"
    gen = FlextInfraBaseMkGenerator()
    result = gen.write(content, output=output_path)
    tm.ok(result)
    assert output_path.exists()


def test_generator_write_to_stream() -> None:
    """Test write() writes to stream when output is None."""
    stream = io.StringIO()
    content = "all:\n\t@echo 'test'\n"
    gen = FlextInfraBaseMkGenerator()
    result = gen.write(content, stream=stream)
    tm.ok(result)
    assert stream.getvalue() == content


def test_generator_write_fails_without_output_or_stream() -> None:
    """Test write() fails when neither output nor stream provided."""
    content = "all:\n\t@echo 'test'\n"
    gen = FlextInfraBaseMkGenerator()
    result = gen.write(content, output=None, stream=None)
    tm.fail(result)
    assert isinstance(result.error, str)
    assert "stdout stream is required" in result.error
