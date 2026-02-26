from __future__ import annotations

from pathlib import Path

from _pytest.capture import CaptureFixture
from flext_core.result import FlextResult as r
from flext_infra.basemk.__main__ import main as basemk_main
from flext_infra.basemk.engine import TemplateEngine
from flext_infra.basemk.generator import BaseMkGenerator
from flext_infra.models import m as im


class _InvalidTemplateEngine:
    def render_all(self, config: im.BaseMkConfig | None = None) -> r[str]:
        del config
        return r[str].ok("ifneq (,\n")


def test_render_all_generates_large_makefile() -> None:
    result = TemplateEngine().render_all()

    assert result.is_success
    content = result.value
    assert len(content.splitlines()) > 400


def test_render_all_has_no_scripts_path_references() -> None:
    result = TemplateEngine().render_all()

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

    result = BaseMkGenerator().generate(config)

    assert result.is_success
    assert "PROJECT_NAME ?= sample-project" in result.value


def test_generator_fails_for_invalid_make_syntax() -> None:
    result = BaseMkGenerator(template_engine=_InvalidTemplateEngine()).generate()

    assert result.is_failure
    assert result.error is not None
    assert "validation failed" in result.error


def test_generator_write_saves_output_file(tmp_path: Path) -> None:
    output_path = tmp_path / "base.mk"
    content = "all:\n\t@true\n"

    result = BaseMkGenerator().write(content, output=output_path)

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
        "# ===================================="
    )
