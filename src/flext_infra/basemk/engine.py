"""Jinja2-based template engine for rendering base.mk configuration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, override

from flext_core.result import r
from flext_core.service import FlextService
from flext_core.typings import t
from jinja2 import (
    Environment,
    FileSystemLoader,
    StrictUndefined,
    TemplateError,
    select_autoescape,
)

from flext_infra.models import m


class TemplateEngine(FlextService[str]):
    """Render base.mk templates with configuration context."""

    TEMPLATE_ORDER: ClassVar[tuple[str, ...]] = (
        "base_header.mk.j2",
        "base_detection.mk.j2",
        "base_venv.mk.j2",
        "base_preflight.mk.j2",
        "base_verbs.mk.j2",
        "base_pr.mk.j2",
        "base_clean.mk.j2",
    )

    def __init__(self) -> None:
        """Initialize the template engine with Jinja2 environment."""
        super().__init__()
        template_root = Path(__file__).resolve().parent / "templates"
        self._environment = Environment(
            loader=FileSystemLoader(str(template_root)),
            trim_blocks=False,
            lstrip_blocks=False,
            keep_trailing_newline=True,
            undefined=StrictUndefined,
            autoescape=select_autoescape(),
        )

    @override
    def execute(self) -> r[str]:
        return self.render_all()

    def render_all(self, config: m.BaseMkConfig | None = None) -> r[str]:
        """Render all base.mk templates in order with the given configuration."""
        active_config = config or self._default_config()
        context: Mapping[str, t.ConfigMapValue] = {
            "config": active_config,
            "lint_gates_csv": ",".join(active_config.lint_gates),
        }
        sections: list[str] = []

        try:
            for template_name in self.TEMPLATE_ORDER:
                template = self._environment.get_template(template_name)
                rendered = template.render(**context)
                sections.append(rendered.rstrip("\n"))
            content = "\n\n".join(sections).rstrip("\n") + "\n"
            return r[str].ok(content)
        except (TemplateError, ValueError, TypeError) as exc:
            return r[str].fail(f"base.mk template render failed: {exc}")

    @staticmethod
    def _default_config() -> m.BaseMkConfig:
        """Return the default base.mk configuration."""
        return m.BaseMkConfig.model_validate({
            "project_name": "unnamed",
            "python_version": "3.13",
            "core_stack": "python",
            "package_manager": "poetry",
            "source_dir": "src",
            "tests_dir": "tests",
            "lint_gates": ["lint", "format", "pyrefly", "mypy", "pyright"],
            "test_command": "pytest",
        })

    @classmethod
    def default_config(cls) -> m.BaseMkConfig:
        """Return the default base.mk configuration."""
        return cls._default_config()


__all__ = ["TemplateEngine"]
