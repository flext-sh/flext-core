"""AST-driven centralizer for Pydantic models and dict-like contracts."""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class _ClassMove:
    name: str
    start: int
    end: int
    source: str
    kind: str


@dataclass(slots=True, frozen=True)
class _AliasMove:
    name: str
    start: int
    end: int
    alias_expr: str


class FlextInfraRefactorPydanticCentralizer:
    """Centralize model-like contracts into `models.py`/`_models.py` files."""

    _SCOPE_DIRS: tuple[str, ...] = ("src", "tests", "scripts", "examples")
    _SKIP_DIRS: tuple[str, ...] = (
        ".git",
        ".venv",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
    )
    _MODEL_BASES: tuple[str, ...] = (
        "BaseModel",
        "RootModel",
        "TypedDict",
        "ArbitraryTypesModel",
        "FrozenModel",
        "FrozenStrictModel",
    )
    _PROTECTED_FILENAMES: tuple[str, ...] = (
        "settings.py",
        "__init__.py",
    )
    _TYPED_DICT_MIN_ARGS: int = 2

    @staticmethod
    def _is_target_python(file_path: Path) -> bool:
        if file_path.suffix != ".py":
            return False
        if any(
            part in FlextInfraRefactorPydanticCentralizer._SKIP_DIRS
            for part in file_path.parts
        ):
            return False
        if file_path.name in FlextInfraRefactorPydanticCentralizer._PROTECTED_FILENAMES:
            return False
        parts = set(file_path.parts)
        return (
            len(parts.intersection(FlextInfraRefactorPydanticCentralizer._SCOPE_DIRS))
            > 0
        )

    @staticmethod
    def _is_allowed_model_path(file_path: Path) -> bool:
        posix = file_path.as_posix()
        return posix.endswith(("/models.py", "/_models.py")) or "/models/" in posix

    @staticmethod
    def _class_base_names(node: ast.ClassDef) -> set[str]:
        names: set[str] = set()
        for base in node.bases:
            if isinstance(base, ast.Name):
                names.add(base.id)
            elif isinstance(base, ast.Attribute):
                names.add(base.attr)
                root = ast.unparse(base)
                if root:
                    names.add(root)
        return names

    @staticmethod
    def _is_model_like_base_name(base_name: str) -> bool:
        if base_name in FlextInfraRefactorPydanticCentralizer._MODEL_BASES:
            return True
        if base_name.startswith("FlextModels."):
            return True
        return base_name.endswith("Model")

    class _DisallowedModelBaseNormalizer(ast.NodeTransformer):
        @staticmethod
        def _base_name(base: ast.expr) -> str:
            if isinstance(base, ast.Name):
                return base.id
            if isinstance(base, ast.Attribute):
                return base.attr
            return ""

        def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
            updated = self.generic_visit(node)
            if not isinstance(updated, ast.ClassDef):
                return updated
            updated.bases = [
                base
                for base in updated.bases
                if self._base_name(base) not in {"BaseModel", "TypedDict"}
            ]
            updated.keywords = [
                kw
                for kw in updated.keywords
                if not (
                    kw.arg == "total"
                    and isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, bool)
                )
            ]
            return updated

    @staticmethod
    def _is_top_level_model_class(node: ast.stmt) -> bool:
        if not isinstance(node, ast.ClassDef):
            return False
        base_names = FlextInfraRefactorPydanticCentralizer._class_base_names(node)
        return any(
            FlextInfraRefactorPydanticCentralizer._is_model_like_base_name(base_name)
            for base_name in base_names
        )

    @staticmethod
    def _is_typings_scope(file_path: Path) -> bool:
        posix = file_path.as_posix()
        return posix.endswith(("/typings.py", "/_typings.py")) or "/typings/" in posix

    @staticmethod
    def _is_dict_like_expr(expr: str) -> bool:
        return any(
            marker in expr for marker in ("dict[", "Mapping[", "MutableMapping[")
        )

    @staticmethod
    def _is_dict_like_alias(
        node: ast.stmt, source: str, *, file_path: Path
    ) -> _AliasMove | None:
        keys = (
            "dict",
            "payload",
            "schema",
            "entry",
            "config",
            "metadata",
            "fixture",
            "case",
        )
        is_typings_scope = FlextInfraRefactorPydanticCentralizer._is_typings_scope(
            file_path
        )
        match node:
            case ast.TypeAlias():
                alias_name = node.name.id
                if (not is_typings_scope) and (
                    not any(token in alias_name.lower() for token in keys)
                ):
                    return None
                expr = ast.get_source_segment(source, node.value)
                if expr is None:
                    return None
                if not FlextInfraRefactorPydanticCentralizer._is_dict_like_expr(expr):
                    return None
                return _AliasMove(
                    name=alias_name,
                    start=node.lineno,
                    end=node.end_lineno or node.lineno,
                    alias_expr=expr,
                )
            case ast.AnnAssign():
                if not isinstance(node.target, ast.Name):
                    return None
                alias_name = node.target.id
                if (not is_typings_scope) and (
                    not any(token in alias_name.lower() for token in keys)
                ):
                    return None
                if node.value is None:
                    return None
                expr = ast.get_source_segment(source, node.value)
                if expr is None:
                    return None
                if not FlextInfraRefactorPydanticCentralizer._is_dict_like_expr(expr):
                    return None
                annotation = ast.get_source_segment(source, node.annotation) or ""
                if ("TypeAlias" not in annotation) and (not is_typings_scope):
                    return None
                return _AliasMove(
                    name=alias_name,
                    start=node.lineno,
                    end=node.end_lineno or node.lineno,
                    alias_expr=expr,
                )
            case ast.Assign():
                if len(node.targets) != 1:
                    return None
                if not isinstance(node.targets[0], ast.Name):
                    return None
                alias_name = node.targets[0].id
                if (not is_typings_scope) and (
                    not any(token in alias_name.lower() for token in keys)
                ):
                    return None
                expr = ast.get_source_segment(source, node.value)
                if expr is None:
                    return None
                if not FlextInfraRefactorPydanticCentralizer._is_dict_like_expr(expr):
                    return None
                return _AliasMove(
                    name=alias_name,
                    start=node.lineno,
                    end=node.end_lineno or node.lineno,
                    alias_expr=expr,
                )
            case _:
                return None

    @staticmethod
    def _typed_dict_factory_model(node: ast.Assign) -> _ClassMove | None:
        if len(node.targets) != 1:
            return None
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            return None
        if not isinstance(node.value, ast.Call):
            return None
        func = node.value.func
        is_typed_dict_factory = False
        if isinstance(func, ast.Name):
            is_typed_dict_factory = func.id == "TypedDict"
        elif isinstance(func, ast.Attribute):
            is_typed_dict_factory = func.attr == "TypedDict"
        if not is_typed_dict_factory:
            return None
        if (
            len(node.value.args)
            < FlextInfraRefactorPydanticCentralizer._TYPED_DICT_MIN_ARGS
        ):
            return None
        field_map_arg = node.value.args[1]
        if not isinstance(field_map_arg, ast.Dict):
            return None
        field_lines: list[str] = []
        total_false = False
        for kw in node.value.keywords:
            if (
                kw.arg == "total"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is False
            ):
                total_false = True
        for key_node, value_node in zip(
            field_map_arg.keys, field_map_arg.values, strict=True
        ):
            if not isinstance(key_node, ast.Constant):
                continue
            key_value = key_node.value
            if not isinstance(key_value, str):
                continue
            annotation = ast.unparse(value_node)
            if total_false:
                field_lines.append(
                    f"    {key_value}: {annotation} | None = Field(default=None)"
                )
            else:
                field_lines.append(f"    {key_value}: {annotation}")
        if len(field_lines) == 0:
            field_lines.append("    pass")
        rendered_fields = "\n".join(field_lines)
        rendered_class = (
            f"class {target.id}(BaseModel):\n"
            '    model_config = ConfigDict(extra="forbid")\n'
            f"{rendered_fields}\n"
        )
        return _ClassMove(
            name=target.id,
            start=node.lineno,
            end=node.end_lineno or node.lineno,
            source=rendered_class,
            kind="typed_dict_factory",
        )

    @staticmethod
    def _typed_dict_total_false(node: ast.ClassDef) -> bool:
        for keyword in node.keywords:
            if keyword.arg == "total" and isinstance(keyword.value, ast.Constant):
                return bool(keyword.value.value) is False
        return False

    @staticmethod
    def _build_model_from_typed_dict(node: ast.ClassDef, source: str) -> str:
        total_false = FlextInfraRefactorPydanticCentralizer._typed_dict_total_false(
            node
        )
        fields: list[str] = []
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                ann = ast.get_source_segment(source, stmt.annotation)
                if ann is None:
                    continue
                if total_false:
                    fields.append(
                        f"    {stmt.target.id}: {ann} | None = Field(default=None)"
                    )
                else:
                    fields.append(f"    {stmt.target.id}: {ann}")
        if len(fields) == 0:
            fields.append("    pass")
        body = "\n".join(fields)
        return f'class {node.name}(BaseModel):\n    model_config = ConfigDict(extra="forbid")\n{body}\n'

    @staticmethod
    def _collect_moves(file_path: Path) -> tuple[list[_ClassMove], list[_AliasMove]]:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        lines = source.splitlines()
        class_moves: list[_ClassMove] = []
        alias_moves: list[_AliasMove] = []
        for stmt in tree.body:
            typed_dict_factory_move = (
                FlextInfraRefactorPydanticCentralizer._typed_dict_factory_model(stmt)
                if isinstance(stmt, ast.Assign)
                else None
            )
            if typed_dict_factory_move is not None:
                class_moves.append(typed_dict_factory_move)
                continue
            if FlextInfraRefactorPydanticCentralizer._is_top_level_model_class(stmt):
                if not isinstance(stmt, ast.ClassDef):
                    continue
                start = stmt.lineno
                end = stmt.end_lineno or stmt.lineno
                snippet = "\n".join(lines[start - 1 : end])
                base_names = FlextInfraRefactorPydanticCentralizer._class_base_names(
                    stmt
                )
                kind = "typed_dict" if "TypedDict" in base_names else "base_model"
                if kind == "typed_dict":
                    snippet = FlextInfraRefactorPydanticCentralizer._build_model_from_typed_dict(
                        stmt, source
                    )
                class_moves.append(
                    _ClassMove(
                        name=stmt.name, start=start, end=end, source=snippet, kind=kind
                    )
                )
                continue
            alias_move = FlextInfraRefactorPydanticCentralizer._is_dict_like_alias(
                stmt, source, file_path=file_path
            )
            if alias_move is not None:
                alias_moves.append(alias_move)
        return (class_moves, alias_moves)

    @staticmethod
    def _dest_import_statement(file_path: Path, names: list[str]) -> str:
        joined = ", ".join(sorted(set(names)))
        if (file_path.parent / "__init__.py").exists():
            return f"from ._models import {joined}"
        return f"from _models import {joined}"

    @staticmethod
    def _dest_typings_import_statement(file_path: Path, names: list[str]) -> str:
        joined = ", ".join(sorted(set(names)))
        if (file_path.parent / "__init__.py").exists():
            return f"from ._typings import {joined}"
        return f"from _typings import {joined}"

    @staticmethod
    def _insert_import(source: str, import_stmt: str) -> str:
        if import_stmt in source:
            return source
        lines = source.splitlines()
        insert_idx = 0
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("from __future__ import"):
                insert_idx = idx + 1
                continue
            if stripped.startswith(("import ", "from ")):
                insert_idx = idx + 1
                continue
            if not stripped and insert_idx > 0:
                continue
            if insert_idx > 0:
                break
        lines.insert(insert_idx, import_stmt)
        return "\n".join(lines) + ("\n" if source.endswith("\n") else "")

    @staticmethod
    def _rewrite_source(
        file_path: Path, class_moves: list[_ClassMove], alias_moves: list[_AliasMove]
    ) -> str:
        source = file_path.read_text(encoding="utf-8")
        lines = source.splitlines()
        ranges = sorted(
            [(m.start, m.end) for m in class_moves]
            + [(a.start, a.end) for a in alias_moves],
            key=operator.itemgetter(0),
            reverse=True,
        )
        for start, end in ranges:
            del lines[start - 1 : end]
        moved_model_names = [m.name for m in class_moves]
        moved_alias_names = [a.name for a in alias_moves]
        updated = "\n".join(lines)
        if len(moved_model_names) > 0:
            import_stmt = FlextInfraRefactorPydanticCentralizer._dest_import_statement(
                file_path, moved_model_names
            )
            updated = FlextInfraRefactorPydanticCentralizer._insert_import(
                updated, import_stmt
            )
        if len(moved_alias_names) > 0:
            typings_import_stmt = (
                FlextInfraRefactorPydanticCentralizer._dest_typings_import_statement(
                    file_path, moved_alias_names
                )
            )
            updated = FlextInfraRefactorPydanticCentralizer._insert_import(
                updated, typings_import_stmt
            )
        if source.endswith("\n") and (not updated.endswith("\n")):
            updated += "\n"
        return updated

    @staticmethod
    def _ensure_dest_header(dest_path: Path) -> str:
        if dest_path.exists():
            return dest_path.read_text(encoding="utf-8")
        return (
            '"""Auto-generated centralized models."""\n\n'
            "from __future__ import annotations\n\n"
            "from pydantic import BaseModel, ConfigDict, Field, RootModel\n\n"
            "class FlextAutoConstants:\n    pass\n\n"
            "class FlextAutoTypes:\n    pass\n\n"
            "class FlextAutoProtocols:\n    pass\n\n"
            "class FlextAutoUtilities:\n    pass\n\n"
            "class FlextAutoModels:\n    pass\n\n"
            "c = FlextAutoConstants\n"
            "t = FlextAutoTypes\n"
            "p = FlextAutoProtocols\n"
            "u = FlextAutoUtilities\n"
            "m = FlextAutoModels\n\n"
        )

    @staticmethod
    def _ensure_typings_header(dest_path: Path) -> str:
        if dest_path.exists():
            return dest_path.read_text(encoding="utf-8")
        return (
            '"""Auto-generated centralized typings."""\n\n'
            "from __future__ import annotations\n\n"
            "from typing import TypeAlias\n\n"
        )

    @staticmethod
    def _append_unique_blocks(
        existing: str, blocks: list[str], names: list[str]
    ) -> str:
        updated = existing
        for name, block in zip(names, blocks, strict=True):
            if f"class {name}(" in updated or f"class {name}:" in updated:
                continue
            updated = updated.rstrip() + "\n\n" + block.rstrip() + "\n"
        return updated

    @staticmethod
    def _alias_as_root_model(alias_move: _AliasMove) -> str:
        return (
            f"class {alias_move.name}(RootModel[{alias_move.alias_expr}]):\n    pass\n"
        )

    @staticmethod
    def _alias_as_type_alias(alias_move: _AliasMove) -> str:
        return f"{alias_move.name}: TypeAlias = {alias_move.alias_expr}\n"

    @staticmethod
    def _normalize_disallowed_bases(file_path: Path, *, apply_changes: bool) -> bool:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        normalizer = (
            FlextInfraRefactorPydanticCentralizer._DisallowedModelBaseNormalizer()
        )
        normalized = normalizer.visit(tree)
        ast.fix_missing_locations(normalized)
        rewritten = ast.unparse(normalized)
        if rewritten == source:
            return False
        if apply_changes:
            file_path.write_text(rewritten + "\n", encoding="utf-8")
        return True

    @staticmethod
    def _scan_file_violations(file_path: Path) -> tuple[int, int]:
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError, OSError):
            return (0, 0)
        model_class_count = 0
        dict_alias_count = 0
        for stmt in tree.body:
            if FlextInfraRefactorPydanticCentralizer._is_top_level_model_class(stmt):
                model_class_count += 1
                continue
            if (
                FlextInfraRefactorPydanticCentralizer._is_dict_like_alias(
                    stmt, source, file_path=file_path
                )
                is not None
            ):
                dict_alias_count += 1
        return (model_class_count, dict_alias_count)

    @staticmethod
    def centralize_workspace(
        workspace_root: Path, *, apply_changes: bool, normalize_remaining: bool
    ) -> dict[str, int]:
        """Centralize model contracts and normalize namespace scaffolds."""
        moved_classes = 0
        moved_aliases = 0
        normalized_files = 0
        touched_files = 0
        scanned_files = 0
        detected_model_violations = 0
        detected_alias_violations = 0
        created_model_files = 0
        created_typings_files = 0
        for file_path in workspace_root.rglob("*.py"):
            if not FlextInfraRefactorPydanticCentralizer._is_target_python(file_path):
                continue
            if FlextInfraRefactorPydanticCentralizer._is_allowed_model_path(file_path):
                continue
            scanned_files += 1
            found_models, found_aliases = (
                FlextInfraRefactorPydanticCentralizer._scan_file_violations(file_path)
            )
            detected_model_violations += found_models
            detected_alias_violations += found_aliases
            try:
                class_moves, alias_moves = (
                    FlextInfraRefactorPydanticCentralizer._collect_moves(file_path)
                )
            except (SyntaxError, UnicodeDecodeError, OSError):
                continue
            if len(class_moves) == 0 and len(alias_moves) == 0:
                continue
            dest_path = file_path.parent / "_models.py"
            class_blocks = [m.source for m in class_moves]
            class_names = [m.name for m in class_moves]
            alias_blocks = [
                FlextInfraRefactorPydanticCentralizer._alias_as_type_alias(a)
                for a in alias_moves
            ]
            alias_names = [a.name for a in alias_moves]
            if not dest_path.exists():
                created_model_files += 1
            existing_dest = FlextInfraRefactorPydanticCentralizer._ensure_dest_header(
                dest_path
            )
            updated_dest = FlextInfraRefactorPydanticCentralizer._append_unique_blocks(
                existing_dest, class_blocks, class_names
            )
            typings_dest_path = file_path.parent / "_typings.py"
            if len(alias_moves) > 0 and not typings_dest_path.exists():
                created_typings_files += 1
            existing_typings_dest = (
                FlextInfraRefactorPydanticCentralizer._ensure_typings_header(
                    typings_dest_path
                )
            )
            updated_typings_dest = (
                FlextInfraRefactorPydanticCentralizer._append_unique_blocks(
                    existing_typings_dest,
                    alias_blocks,
                    alias_names,
                )
            )
            updated_source = FlextInfraRefactorPydanticCentralizer._rewrite_source(
                file_path, class_moves, alias_moves
            )
            moved_classes += len(class_moves)
            moved_aliases += len(alias_moves)
            touched_files += 1
            if apply_changes:
                dest_path.write_text(updated_dest, encoding="utf-8")
                if len(alias_moves) > 0:
                    typings_dest_path.write_text(updated_typings_dest, encoding="utf-8")
                file_path.write_text(updated_source, encoding="utf-8")
        if normalize_remaining:
            for file_path in workspace_root.rglob("*.py"):
                if not FlextInfraRefactorPydanticCentralizer._is_target_python(
                    file_path
                ):
                    continue
                if FlextInfraRefactorPydanticCentralizer._is_allowed_model_path(
                    file_path
                ):
                    continue
                try:
                    changed = FlextInfraRefactorPydanticCentralizer._normalize_disallowed_bases(
                        file_path, apply_changes=apply_changes
                    )
                except (SyntaxError, UnicodeDecodeError, OSError):
                    continue
                if changed:
                    normalized_files += 1
        return {
            "scanned_files": scanned_files,
            "touched_files": touched_files,
            "moved_classes": moved_classes,
            "moved_aliases": moved_aliases,
            "normalized_files": normalized_files,
            "detected_model_violations": detected_model_violations,
            "detected_alias_violations": detected_alias_violations,
            "created_model_files": created_model_files,
            "created_typings_files": created_typings_files,
        }
