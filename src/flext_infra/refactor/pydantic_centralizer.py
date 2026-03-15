"""AST-driven centralizer for Pydantic models and dict-like contracts."""

from __future__ import annotations

from pathlib import Path

from flext_infra import m, u
from flext_infra.refactor.analysis import FlextInfraRefactorPydanticCentralizerAnalysis


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
    _PROTECTED_FILENAMES: tuple[str, ...] = (
        "settings.py",
        "__init__.py",
    )
    _AUTO_APPLY_CLASS_KINDS: tuple[str, ...] = ("typed_dict", "typed_dict_factory")

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
    def _dest_import_statement(file_path: Path, names: list[str]) -> str:
        joined = ", ".join(sorted(set(names)))
        if (file_path.parent / "__init__.py").exists():
            return f"from ._models import {joined}"
        return f"from _models import {joined}"

    @staticmethod
    def _ensure_dest_header(dest_path: Path) -> str:
        if dest_path.exists():
            return dest_path.read_text(encoding="utf-8")
        return (
            '"""Auto-generated centralized models."""\n\n'
            "from __future__ import annotations\n\n"
            "from typing import Annotated\n\n"
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
        existing: str,
        blocks: list[str],
        names: list[str],
    ) -> str:
        updated = existing
        for name, block in zip(names, blocks, strict=True):
            if f"class {name}(" in updated or f"class {name}:" in updated:
                continue
            updated = updated.rstrip() + "\n\n" + block.rstrip() + "\n"
        return updated

    @staticmethod
    def _alias_as_root_model(alias_move: m.Infra.Refactor.AliasMove) -> str:
        return (
            f"class {alias_move.name}(RootModel[{alias_move.alias_expr}]):\n    pass\n"
        )

    @staticmethod
    def _normalize_disallowed_bases(file_path: Path, *, apply: bool) -> bool:
        _ = file_path
        _ = apply
        return False

    @staticmethod
    def _can_apply_import_rewrite(file_path: Path) -> bool:
        return (file_path.parent / "__init__.py").exists()

    @staticmethod
    def _filter_moves_for_necessity(
        class_moves: list[m.Infra.Refactor.ClassMove],
        alias_moves: list[m.Infra.Refactor.AliasMove],
    ) -> tuple[list[m.Infra.Refactor.ClassMove], list[m.Infra.Refactor.AliasMove]]:
        filtered_classes = [
            move
            for move in class_moves
            if move.kind
            in FlextInfraRefactorPydanticCentralizer._AUTO_APPLY_CLASS_KINDS
        ]
        return (filtered_classes, alias_moves)

    @staticmethod
    def centralize_workspace(
        workspace_root: Path,
        *,
        apply: bool,
        normalize_remaining: bool,
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
        skipped_nonpackage_apply = 0
        skipped_non_necessary_apply = 0
        failure_stats = m.Infra.Refactor.CentralizerFailureStats()
        files_result = u.Infra.iter_python_files(workspace_root=workspace_root)
        if files_result.is_failure:
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
                "skipped_nonpackage_apply": skipped_nonpackage_apply,
                "skipped_non_necessary_apply": skipped_non_necessary_apply,
                "parse_syntax_errors": failure_stats.parse_syntax_errors,
                "parse_encoding_errors": failure_stats.parse_encoding_errors,
                "parse_io_errors": failure_stats.parse_io_errors + 1,
            }
        python_files = files_result.value
        for file_path in python_files:
            if not FlextInfraRefactorPydanticCentralizer._is_target_python(file_path):
                continue
            if FlextInfraRefactorPydanticCentralizer._is_allowed_model_path(file_path):
                continue
            scanned_files += 1
            found_models, found_aliases = (
                FlextInfraRefactorPydanticCentralizerAnalysis.scan_file_violations(
                    file_path,
                )
            )
            detected_model_violations += found_models
            detected_alias_violations += found_aliases
            collected_moves = (
                FlextInfraRefactorPydanticCentralizerAnalysis.collect_moves_safe(
                    file_path,
                    failure_stats=failure_stats,
                )
            )
            if collected_moves is None:
                continue
            class_moves, alias_moves = collected_moves
            if len(class_moves) == 0 and len(alias_moves) == 0:
                continue
            apply_class_moves = class_moves
            apply_alias_moves = alias_moves
            if apply:
                apply_class_moves, apply_alias_moves = (
                    FlextInfraRefactorPydanticCentralizer._filter_moves_for_necessity(
                        class_moves,
                        alias_moves,
                    )
                )
                if len(apply_class_moves) == 0 and len(apply_alias_moves) == 0:
                    skipped_non_necessary_apply += 1
                    continue
            dest_path = file_path.parent / "_models.py"
            class_blocks = [m.source for m in apply_class_moves]
            class_names = [m.name for m in apply_class_moves]
            alias_blocks = [
                FlextInfraRefactorPydanticCentralizer._alias_as_root_model(a)
                for a in apply_alias_moves
            ]
            alias_names = [a.name for a in apply_alias_moves]
            if not dest_path.exists():
                created_model_files += 1
            existing_dest = FlextInfraRefactorPydanticCentralizer._ensure_dest_header(
                dest_path,
            )
            updated_dest = FlextInfraRefactorPydanticCentralizer._append_unique_blocks(
                existing_dest,
                class_blocks,
                class_names,
            )
            updated_dest = FlextInfraRefactorPydanticCentralizer._append_unique_blocks(
                updated_dest,
                alias_blocks,
                alias_names,
            )
            moved_names = [m.name for m in apply_class_moves] + [
                a.name for a in apply_alias_moves
            ]
            updated_source = FlextInfraRefactorPydanticCentralizerAnalysis.rewrite_source(
                file_path,
                apply_class_moves,
                apply_alias_moves,
                import_statement=FlextInfraRefactorPydanticCentralizer._dest_import_statement(
                    file_path,
                    moved_names,
                ),
            )
            moved_classes += len(apply_class_moves)
            moved_aliases += len(apply_alias_moves)
            touched_files += 1
            if apply:
                if not FlextInfraRefactorPydanticCentralizer._can_apply_import_rewrite(
                    file_path,
                ):
                    skipped_nonpackage_apply += 1
                    continue
                _ = dest_path.write_text(updated_dest, encoding="utf-8")
                _ = file_path.write_text(updated_source, encoding="utf-8")
        if normalize_remaining:
            for file_path in python_files:
                if not FlextInfraRefactorPydanticCentralizer._is_target_python(
                    file_path,
                ):
                    continue
                if FlextInfraRefactorPydanticCentralizer._is_allowed_model_path(
                    file_path,
                ):
                    continue
                try:
                    changed = FlextInfraRefactorPydanticCentralizer._normalize_disallowed_bases(
                        file_path,
                        apply=apply,
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
            "skipped_nonpackage_apply": skipped_nonpackage_apply,
            "skipped_non_necessary_apply": skipped_non_necessary_apply,
            "parse_syntax_errors": failure_stats.parse_syntax_errors,
            "parse_encoding_errors": failure_stats.parse_encoding_errors,
            "parse_io_errors": failure_stats.parse_io_errors,
        }
