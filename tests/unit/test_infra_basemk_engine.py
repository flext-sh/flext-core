"""Tests for FlextBasemkEngine to achieve full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _pytest.capture import CaptureFixture
from flext_core import FlextResult as r
from flext_infra import m as im
from flext_infra.basemk.__main__ import main as basemk_main
from flext_infra.basemk.engine import FlextInfraBaseMkTemplateEngine
from flext_infra.basemk.generator import FlextInfraBaseMkGenerator
from jinja2 import TemplateError


class _InvalidTemplateEngine:
    def render_all(self, config: im.BaseMkConfig | None = None) -> r[str]:
        del config
        return r[str].ok("ifneq (,\n")


def test_render_all_generates_large_makefile() -> None:
    result = FlextInfraBaseMkTemplateEngine().render_all()

    assert result.is_success
    content = result.value
    assert len(content.splitlines()) > 400


def test_render_all_has_no_scripts_path_references() -> None:
    result = FlextInfraBaseMkTemplateEngine().render_all()

    assert result.is_success
    assert "scripts/" not in result.value


def test_generator_renders_with_config_override() -> None:
    config = im.BaseMkConfig.model_validate({
        "project_name": "sample-project",
        "python_version": "3.11",
        "core_stack": "python",
        "package_manager": "pip",
        "source_dir": "lib",
        "tests_dir": "tests/unit",
        "lint_gates": ["mypy", "ruff"],
        "test_command": "tox",
    })
    config = im.BaseMkConfig(
        project_name="sample-project",
        python_version="3.13",
        core_stack="python",
        package_manager="poetry",
        source_dir="src",
        tests_dir="tests",
        lint_gates=["lint", "mypy"],
        test_command="pytest",
    )

    result = FlextInfraBaseMkGenerator().generate(config)

    assert result.is_success
    assert "PROJECT_NAME ?= sample-project" in result.value


def test_generator_fails_for_invalid_make_syntax() -> None:
    result = FlextInfraBaseMkGenerator(
        template_engine=_InvalidTemplateEngine()
    ).generate()

    assert result.is_failure
    assert result.error is not None
    assert "validation failed" in result.error


def test_generator_write_saves_output_file(tmp_path: Path) -> None:
    output_path = tmp_path / "base.mk"
    content = "all:\n\t@true\n"

    result = FlextInfraBaseMkGenerator().write(content, output=output_path)

    assert result.is_success
    assert output_path.read_text(encoding="utf-8") == content


def test_basemk_cli_generate_to_stdout(capsys: CaptureFixture[str]) -> None:
    exit_code = basemk_main(["generate", "--project-name", "cli-project"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PROJECT_NAME ?= cli-project" in captured.out


def test_basemk_cli_generate_to_file(tmp_path: Path) -> None:
    output_path = tmp_path / "base.mk"

    exit_code = basemk_main(["generate", "--output", str(output_path)])

    assert exit_code == 0
    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8").startswith(
        "# ====================================",
    )


def test_basemk_engine_render_all_returns_string() -> None:
    """Test engine.render_all() returns string."""
    # FlextInfraBaseMkTemplateEngine is the correct class

    engine = FlextInfraBaseMkTemplateEngine()
    result = engine.render_all()
    assert result.is_success
    assert isinstance(result.value, str) and len(result.value) > 0


def test_basemk_engine_render_all_with_valid_config() -> None:
    """Test engine.render_all() with explicit config."""
    # im is already imported at the top

    config = im.BaseMkConfig(
        project_name="test-project",
        python_version="3.13",
        core_stack="python",
        package_manager="poetry",
        source_dir="src",
        tests_dir="tests",
        lint_gates=["mypy", "ruff"],
        test_command="pytest",
    )
    engine = FlextInfraBaseMkTemplateEngine()
    result = engine.render_all(config=config)
    assert result.is_success
    assert "PROJECT_NAME ?= test-project" in result.value


def test_basemk_engine_execute_calls_render_all() -> None:
    """Test engine.execute() calls render_all()."""
    engine = FlextInfraBaseMkTemplateEngine()
    result = engine.execute()
    assert result.is_success
    assert isinstance(result.value, str)
    assert len(result.value) > 0


def test_basemk_engine_render_all_handles_template_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test engine.render_all() handles TemplateError gracefully."""

    def mock_get_template(*args: object, **kwargs: object) -> object:
        msg = "Template not found"
        raise TemplateError(msg)

    engine = FlextInfraBaseMkTemplateEngine()
    monkeypatch.setattr(engine._environment, "get_template", mock_get_template)
    result = engine.render_all()
    assert result.is_failure
    assert "template render failed" in result.error
