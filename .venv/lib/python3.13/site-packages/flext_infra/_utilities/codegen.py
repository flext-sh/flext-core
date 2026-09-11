"""Codegen utilities composition for the infrastructure namespace."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from flext_cli import u
from flext_infra.constants import c
from flext_infra.models import m
from flext_infra.protocols import p
from flext_infra.typings import t


class FlextInfraUtilitiesCodegen:
    """Compose all codegen utility concerns for ``u.Infra``."""

    if TYPE_CHECKING:

        @staticmethod
        def project_root(file_path: Path) -> Path | None: ...

    @staticmethod
    def generate_module_skeleton(
        *, class_name: str, base_class: str, base_module: str, docstring: str
    ) -> str:
        """Render one module skeleton through the cli template engine (ADR-005).

        The body lives in ``templates/module_skeleton.py.j2``; this method only
        builds the context (explicit base module) and renders fail-closed via
        ``u.Cli.template_render``. A render failure is a real incident and
        surfaces via ``unwrap`` (no silent fallback).
        """
        template_path = (
            Path(__file__).resolve().parent.parent
            / "templates"
            / c.Infra.TEMPLATE_MODULE_SKELETON
        )
        # NOTE (multi-agent, flext-wkii.17 / agent: uv_overlay_owner): preserve
        # the exact validated model identity across the template boundary.
        context = m.Infra.ModuleSkeletonRenderContext(
            class_name=class_name,
            base_class=base_class,
            base_module=base_module,
            docstring=docstring,
        )
        rendered: p.Result[str] = u.Cli.template_render(template_path, context)
        content: str = rendered.unwrap()
        return content

    @staticmethod
    def dir_has_py_files(pkg_dir: Path) -> bool:
        """Return whether a package directory contains canonical Python files."""
        if not pkg_dir.is_dir():
            return False
        return any(
            child.is_file() and child.suffix == ".py" for child in pkg_dir.iterdir()
        )

    @staticmethod
    def parse_final_constant_definitions(
        source_lines: t.SequenceOf[str],
    ) -> t.SequenceOf[tuple[str, str, str, str, int]]:
        """Parse ``NAME: Final[...] = VALUE`` definitions with class-path context."""
        class_stack: t.MutableSequenceOf[tuple[str, int]] = []
        parsed: t.MutableSequenceOf[tuple[str, str, str, str, int]] = []
        for line_number, line in enumerate(source_lines, 1):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            FlextInfraUtilitiesCodegen.update_class_stack(class_stack, stripped, indent)
            match = c.Infra.DETECTION_FINAL_DECL_RE.match(line)
            if match is None:
                continue
            parsed.append((
                match.group("name"),
                match.group("ann"),
                match.group("value").strip(),
                ".".join(name for name, _ in class_stack),
                line_number,
            ))
        return tuple(parsed)

    @staticmethod
    def update_class_stack(
        class_stack: t.MutableSequenceOf[tuple[str, int]],
        stripped_line: str,
        indent: int,
    ) -> None:
        """Keep class-path stack in sync while iterating constant source lines."""
        class_match = (
            c.Infra.DETECTION_CLASS_DECL_RE.match(stripped_line)
            if stripped_line.startswith("class ") and stripped_line.endswith(":")
            else None
        )
        if class_match is not None:
            while class_stack and class_stack[-1][1] >= indent:
                class_stack.pop()
            class_stack.append((class_match.group(1), indent))
            return
        while class_stack and indent <= class_stack[-1][1]:
            class_stack.pop()


__all__: list[str] = ["FlextInfraUtilitiesCodegen"]
