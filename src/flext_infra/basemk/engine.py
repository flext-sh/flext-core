"""Jinja2-based template engine for rendering base.mk configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, override

from jinja2 import (
    Environment,
    FileSystemLoader,
    StrictUndefined,
    TemplateError,
    select_autoescape,
)

from flext_core import r, s
from flext_infra import c, m


class _TemplateRenderer(Protocol):
    def render(
        self, *, config: m.Infra.Basemk.BaseMkConfig, lint_gates_csv: str
    ) -> str: ...


class FlextInfraBaseMkTemplateEngine(s[str]):
    """Render base.mk templates with configuration context."""

    def __init__(self) -> None:
        """Initialize the template engine with Jinja2 environment."""
        super().__init__(
            config_type=None,
            config_overrides=None,
            initial_context=None,
            subproject=None,
            services=None,
            factories=None,
            resources=None,
            container_overrides=None,
            wire_modules=None,
            wire_packages=None,
            wire_classes=None,
        )
        template_root = Path(__file__).resolve().parent / "templates"
        self._environment = Environment(
            loader=FileSystemLoader(str(template_root)),
            trim_blocks=False,
            lstrip_blocks=False,
            keep_trailing_newline=True,
            undefined=StrictUndefined,
            autoescape=select_autoescape(),
        )

    @classmethod
    def default_config(cls) -> m.Infra.Basemk.BaseMkConfig:
        """Return the default base.mk configuration."""
        return cls._default_config()

    @staticmethod
    def _default_config() -> m.Infra.Basemk.BaseMkConfig:
        """Return the default base.mk configuration."""
        return m.Infra.Basemk.BaseMkConfig(
            project_name=c.Infra.Defaults.UNNAMED,
            python_version="3.13",
            core_stack=c.Infra.Toml.PYTHON,
            package_manager=c.Infra.Toml.POETRY,
            source_dir=c.Infra.Paths.DEFAULT_SRC_DIR,
            tests_dir=c.Infra.Directories.TESTS,
            lint_gates=[
                c.Infra.Gates.LINT,
                c.Infra.Gates.FORMAT,
                c.Infra.Gates.PYREFLY,
                c.Infra.Gates.MYPY,
                c.Infra.Gates.PYRIGHT,
            ],
            test_command=c.Infra.Toml.PYTEST,
        )

    @staticmethod
    def _render_template(
        template: _TemplateRenderer,
        config: m.Infra.Basemk.BaseMkConfig,
        lint_gates_csv: str,
    ) -> str:
        return template.render(config=config, lint_gates_csv=lint_gates_csv)

    @override
    def execute(self) -> r[str]:
        return self.render_all()

    def render_all(self, config: m.Infra.Basemk.BaseMkConfig | None = None) -> r[str]:
        """Render all base.mk templates in order with the given configuration."""
        active_config = config or self._default_config()
        lint_gates_csv = ",".join(active_config.lint_gates)
        sections: list[str] = []
        try:
            for template_name in c.Infra.Basemk.TEMPLATE_ORDER:
                template: _TemplateRenderer = self._environment.get_template(
                    template_name,
                )
                rendered = self._render_template(
                    template,
                    active_config,
                    lint_gates_csv,
                )
                sections.append(rendered.rstrip("\n"))
            content = "\n\n".join(sections).rstrip("\n") + "\n"
            return r[str].ok(content)
        except (TemplateError, ValueError, TypeError) as exc:
            return r[str].fail(f"base.mk template render failed: {exc}")


__all__ = ["FlextInfraBaseMkTemplateEngine"]
