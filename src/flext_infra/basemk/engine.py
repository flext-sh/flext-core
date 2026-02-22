from __future__ import annotations

from pathlib import Path
from typing import ClassVar, override

from flext_core.result import FlextResult as r
from flext_core.service import FlextService
from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateError

from flext_infra.models import im


class TemplateEngine(FlextService[str]):
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
        super().__init__()
        template_root = Path(__file__).resolve().parent / "templates"
        self._environment = Environment(
            loader=FileSystemLoader(str(template_root)),
            trim_blocks=False,
            lstrip_blocks=False,
            keep_trailing_newline=True,
            undefined=StrictUndefined,
            autoescape=False,
        )

    @override
    def execute(self) -> r[str]:
        return self.render_all()

    def render_all(self, config: im.BaseMkConfig | None = None) -> r[str]:
        active_config = config or self._default_config()
        context: dict[str, object] = {
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
    def _default_config() -> im.BaseMkConfig:
        return im.BaseMkConfig(
            project_name="unnamed",
            python_version="3.13",
            core_stack="python",
            package_manager="poetry",
            source_dir="src",
            tests_dir="tests",
            lint_gates=["lint", "format", "pyrefly", "mypy", "pyright"],
            test_command="pytest",
        )

    @classmethod
    def default_config(cls) -> im.BaseMkConfig:
        return cls._default_config()


__all__ = ["TemplateEngine"]
