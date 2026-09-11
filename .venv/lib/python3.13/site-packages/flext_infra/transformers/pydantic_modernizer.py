"""Pydantic modernizer transformer — migrate v1/legacy patterns to Pydantic v2.

Conservative AST rewrites:

- ``class Config:`` inside ``BaseModel`` → ``model_config = ConfigDict(...)``.
- ``.dict(...)`` → ``.model_dump(...)``.
- ``.json(...)`` → ``.model_dump_json(...)``.
- ``.parse_obj(...)`` → ``.model_validate(...)``.
- ``.schema(...)`` → ``.model_json_schema(...)``.
- ``@validator`` → ``@field_validator`` with ``mode="before"/"after"``.
- ``@root_validator`` → ``@model_validator`` with ``mode="before"/"after"``.
- ``__fields__`` / ``__field_defaults__`` → ``model_fields`` / ``model_config``.

The transformer only rewrites when the result is unambiguous.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, ClassVar, override

from flext_infra.transformers._rewrite import FlextInfraSourceRewriter
from flext_infra.transformers.base import FlextInfraRopeTransformer

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraRefactorPydanticModernizer(FlextInfraRopeTransformer):
    """AST-driven transformer for Pydantic v2 migration."""

    _description = "migrate Pydantic v1/legacy patterns to v2 canonical forms"

    _PYDANTIC_BASES: ClassVar[frozenset[str]] = frozenset({
        "BaseModel",
        "BaseSettings",
        "RootModel",
    })
    _CONFIG_MIN_BODY_LINES: ClassVar[int] = 2

    @override
    def apply_to_source(self, source: str) -> t.Infra.TransformResult:
        """Apply Pydantic modernizations to source text."""
        self.changes.clear()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source, list(self.changes)

        visitor = self._PydanticVisitor(source)
        visitor.visit(tree)

        if not visitor.rewrites:
            return source, list(self.changes)

        updated = FlextInfraSourceRewriter.apply_rewrites(source, visitor.rewrites)
        for change in visitor.changes:
            self._record_change(change)

        return updated, list(self.changes)

    class _PydanticVisitor(FlextInfraSourceRewriter):
        """Collect rewrites for Pydantic anti-patterns."""

        def __init__(self, source: str) -> None:
            super().__init__(source)
            self._needs_config_dict_import = False
            self._needs_model_validator_import = False
            self._needs_field_validator_import = False

        @override
        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            """Detect ``class Config:`` inside Pydantic models."""
            if self._is_pydantic_model(node):
                for item in node.body:
                    if isinstance(item, ast.ClassDef) and item.name == "Config":
                        self._rewrite_config_class(item)
            self.generic_visit(node)

        @override
        def visit_Call(self, node: ast.Call) -> None:
            """Modernize Pydantic method calls and validator decorators."""
            if isinstance(node.func, ast.Attribute):
                self._rewrite_method_call(node)
            self.generic_visit(node)

        @override
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            """Modernize validator decorators."""
            self._rewrite_validator_decorators(node)
            self.generic_visit(node)

        @override
        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            """Modernize validator decorators on async functions."""
            self._rewrite_validator_decorators(node)
            self.generic_visit(node)

        @override
        def visit_Attribute(self, node: ast.Attribute) -> None:
            """Modernize dunder field access on model classes."""
            if node.attr in {"__fields__", "__field_defaults__"}:
                new_attr = (
                    "model_fields" if node.attr == "__fields__" else "model_config"
                )
                attr_text = self.node_text(node)
                new_text = attr_text[: -len(node.attr)] + new_attr
                self.append_rewrite(
                    node, new_text, f"Replaced {node.attr} with {new_attr}"
                )
            self.generic_visit(node)

        def _is_pydantic_model(self, node: ast.ClassDef) -> bool:
            """Return whether a class definition inherits from a Pydantic base."""
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id in self._model_bases():
                    return True
                if (
                    isinstance(base, ast.Subscript)
                    and isinstance(base.value, ast.Name)
                    and base.value.id in self._model_bases()
                ):
                    return True
            return False

        @staticmethod
        def _model_bases() -> frozenset[str]:
            """Return recognized Pydantic base class names."""
            return FlextInfraRefactorPydanticModernizer._PYDANTIC_BASES

        def _rewrite_config_class(self, node: ast.ClassDef) -> None:
            """Convert ``class Config:`` body into ``model_config = ConfigDict(...)``."""
            body_text = self.node_text(node)
            lines = body_text.splitlines(keepends=True)
            if len(lines) < FlextInfraRefactorPydanticModernizer._CONFIG_MIN_BODY_LINES:
                return

            # Strip class declaration and dedent body lines.
            body_lines = lines[1:]
            first_stripped = body_lines[0].lstrip()
            indent = len(body_lines[0]) - len(first_stripped)
            dedented = [line[indent:] for line in body_lines]

            # Extract assignments, skipping docstrings and non-assignments.
            assignments: list[str] = []
            for line in dedented:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "=" in stripped and not stripped.startswith(('"""', "'''")):
                    assignments.append(stripped)

            if not assignments:
                return

            config_dict_body = ", ".join(assignments)
            class_indent = lines[0][: len(lines[0]) - len(lines[0].lstrip())]
            new_text = f"{class_indent}model_config = ConfigDict({config_dict_body})\n"
            self.append_rewrite(node, new_text, "Converted Config class to ConfigDict")
            self._needs_config_dict_import = True

        def _rewrite_method_call(self, node: ast.Call) -> None:
            """Modernize Pydantic method calls on instances."""
            if not isinstance(node.func, ast.Attribute):
                return
            attr = node.func.attr
            mapping = {
                "dict": "model_dump",
                "json": "model_dump_json",
                "parse_obj": "model_validate",
                "schema": "model_json_schema",
                "schema_json": "model_json_schema",
                "copy": "model_copy",
                "parse_raw": "model_validate_json",
            }
            if attr not in mapping:
                return

            call_text = self.node_text(node)
            new_call = call_text.replace(f".{attr}(", f".{mapping[attr]}(", 1)
            self.append_rewrite(
                node, new_call, f"Replaced .{attr}() with .{mapping[attr]}()"
            )

        def _rewrite_validator_decorators(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            """Convert @validator / @root_validator decorators."""
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                if not isinstance(decorator.func, ast.Name):
                    continue

                name = decorator.func.id
                if name == "validator":
                    self._rewrite_validator_call(decorator, node)
                elif name == "root_validator":
                    self._rewrite_root_validator_call(decorator, node)

        def _rewrite_validator_call(
            self, decorator: ast.Call, _func: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            """Convert ``@validator("field")`` to ``@field_validator("field")``."""
            dec_text = self.node_text(decorator)
            new_text = dec_text.replace("validator(", "field_validator(", 1)

            # Add mode="before" if pre=True is present, otherwise mode="after".
            if "pre=True" in new_text and "mode=" not in new_text:
                new_text = new_text.replace(")", ', mode="before")', 1)
            elif "mode=" not in new_text:
                new_text = new_text.replace(")", ', mode="after")', 1)

            self.append_rewrite(
                decorator, new_text, "Replaced @validator with @field_validator"
            )
            self._needs_field_validator_import = True

        def _rewrite_root_validator_call(
            self, decorator: ast.Call, _func: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            """Convert ``@root_validator`` to ``@model_validator``."""
            dec_text = self.node_text(decorator)
            new_text = dec_text.replace("root_validator(", "model_validator(", 1)

            if "pre=True" in new_text and "mode=" not in new_text:
                new_text = new_text.replace(")", ', mode="before")', 1)
            elif "mode=" not in new_text:
                new_text = new_text.replace(")", ', mode="after")', 1)

            self.append_rewrite(
                decorator, new_text, "Replaced @root_validator with @model_validator"
            )
            self._needs_model_validator_import = True


__all__: list[str] = ["FlextInfraRefactorPydanticModernizer"]
