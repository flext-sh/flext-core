from __future__ import annotations

from pathlib import Path
from typing import TypedDict, cast

import libcst as cst
import yaml
from libcst.metadata import MetadataWrapper

from flext_infra.refactor.result import FlextInfraRefactorResult
from flext_infra.refactor.transformers.class_nesting import ClassNestingTransformer
from flext_infra.refactor.transformers.helper_consolidation import (
    HelperConsolidationTransformer,
)
from flext_infra.refactor.transformers.nested_class_propagation import (
    NestedClassPropagationTransformer,
)

RefactorResult = FlextInfraRefactorResult


class _MappingEntry(TypedDict, total=False):
    loose_name: str
    helper_name: str
    current_file: str
    target_namespace: str
    target_name: str
    confidence: str


class ClassNestingRefactorRule:
    _CONFIDENCE_RANKS: dict[str, int] = {
        "low": 0,
        "medium": 1,
        "high": 2,
    }

    def __init__(self, config_path: Path | None = None) -> None:
        self._config_path = config_path or Path(__file__).with_name(
            "class-nesting-mappings.yml"
        )

    def apply(self, file_path: Path, dry_run: bool = False) -> RefactorResult:
        try:
            if file_path.suffix != ".py":
                return RefactorResult(
                    file_path=file_path,
                    success=True,
                    modified=False,
                    changes=["Skipped non-Python file"],
                    refactored_code=None,
                )

            source = file_path.read_text(encoding="utf-8")
            tree = cst.parse_module(source)
            mappings = self._load_config()
            confidence_threshold = self._confidence_threshold(mappings)

            class_mappings = self._class_nesting_mappings(
                mappings,
                file_path,
                confidence_threshold,
            )
            helper_mappings = self._helper_consolidation_mappings(
                mappings,
                file_path,
                confidence_threshold,
            )
            class_renames = self._class_rename_mappings(
                mappings,
                file_path,
                confidence_threshold,
            )

            changes: list[str] = []
            tree = self._apply_class_nesting(tree, class_mappings, changes)
            tree = self._apply_helper_consolidation(tree, helper_mappings, changes)
            tree = self._apply_nested_class_propagation(tree, class_renames, changes)

            result_code = tree.code
            modified = result_code != source
            if modified and not dry_run:
                file_path.write_text(result_code, encoding="utf-8")

            return RefactorResult(
                file_path=file_path,
                success=True,
                modified=modified,
                changes=changes,
                refactored_code=result_code,
            )
        except Exception as exc:
            return RefactorResult(
                file_path=file_path,
                success=False,
                modified=False,
                error=str(exc),
                changes=[],
                refactored_code=None,
            )

    def _load_config(self) -> dict[str, object]:
        raw = self._config_path.read_text(encoding="utf-8")
        loaded = cast("object", yaml.safe_load(raw))
        if isinstance(loaded, dict):
            return cast("dict[str, object]", loaded)
        return {}

    def _confidence_threshold(self, config: dict[str, object]) -> str:
        raw = config.get("confidence_threshold", "low")
        if isinstance(raw, str):
            candidate = raw.strip().lower()
            if candidate in self._CONFIDENCE_RANKS:
                return candidate
        return "low"

    def _confidence_allowed(self, confidence: str, threshold: str) -> bool:
        confidence_rank = self._CONFIDENCE_RANKS.get(confidence.strip().lower(), 0)
        threshold_rank = self._CONFIDENCE_RANKS.get(threshold, 0)
        return confidence_rank >= threshold_rank

    def _class_nesting_mappings(
        self,
        config: dict[str, object],
        file_path: Path,
        confidence_threshold: str,
    ) -> dict[str, str]:
        mappings: dict[str, str] = {}
        for entry in self._entries_for_file(
            config.get("class_nesting", []),
            file_path,
            confidence_threshold,
        ):
            loose_name = entry.get("loose_name")
            target_namespace = entry.get("target_namespace")
            if isinstance(loose_name, str) and isinstance(target_namespace, str):
                mappings[loose_name] = target_namespace
        return mappings

    def _helper_consolidation_mappings(
        self,
        config: dict[str, object],
        file_path: Path,
        confidence_threshold: str,
    ) -> dict[str, str]:
        mappings: dict[str, str] = {}
        for entry in self._entries_for_file(
            config.get("helper_consolidation", []),
            file_path,
            confidence_threshold,
        ):
            helper_name = entry.get("helper_name")
            target_namespace = entry.get("target_namespace")
            if isinstance(helper_name, str) and isinstance(target_namespace, str):
                mappings[helper_name] = target_namespace
        return mappings

    def _class_rename_mappings(
        self,
        config: dict[str, object],
        file_path: Path,
        confidence_threshold: str,
    ) -> dict[str, str]:
        mappings: dict[str, str] = {}
        for entry in self._entries_for_file(
            config.get("class_nesting", []),
            file_path,
            confidence_threshold,
        ):
            loose_name = entry.get("loose_name")
            target_namespace = entry.get("target_namespace")
            target_name = entry.get("target_name")
            if not isinstance(loose_name, str):
                continue
            if not isinstance(target_namespace, str):
                continue
            if not isinstance(target_name, str):
                continue
            mappings[loose_name] = f"{target_namespace}.{target_name}"
        return mappings

    def _entries_for_file(
        self,
        raw_entries: object,
        file_path: Path,
        confidence_threshold: str,
    ) -> list[_MappingEntry]:
        entries = self._to_object_list(raw_entries)
        if not entries:
            return []

        module_path = self._normalize_module_path(file_path)
        accepted: list[_MappingEntry] = []
        for raw_entry in entries:
            entry = self._coerce_entry(raw_entry)
            if entry is None:
                continue
            current_file = entry.get("current_file")
            if current_file is None:
                continue

            current_module = self._normalize_module_path(Path(current_file))
            if current_module != module_path:
                continue

            confidence = entry.get("confidence", "low")
            if not self._confidence_allowed(confidence, confidence_threshold):
                continue

            accepted.append(entry)
        return accepted

    def _to_object_list(self, value: object) -> list[object]:
        if not isinstance(value, list):
            return []
        return cast(list[object], value)

    def _coerce_entry(self, raw_entry: object) -> _MappingEntry | None:
        if not isinstance(raw_entry, dict):
            return None
        typed = cast(dict[str, object], raw_entry)

        current_file = typed.get("current_file")
        confidence = typed.get("confidence")
        if not isinstance(current_file, str):
            return None
        if confidence is not None and not isinstance(confidence, str):
            return None

        entry: _MappingEntry = {"current_file": current_file}
        if isinstance(typed.get("loose_name"), str):
            entry["loose_name"] = cast("str", typed["loose_name"])
        if isinstance(typed.get("helper_name"), str):
            entry["helper_name"] = cast("str", typed["helper_name"])
        if isinstance(typed.get("target_namespace"), str):
            entry["target_namespace"] = cast("str", typed["target_namespace"])
        if isinstance(typed.get("target_name"), str):
            entry["target_name"] = cast("str", typed["target_name"])
        if isinstance(confidence, str):
            entry["confidence"] = confidence
        return entry

    def _normalize_module_path(self, path_value: Path) -> str:
        normalized = path_value.as_posix().replace("\\", "/")
        path = Path(normalized)
        parts = path.parts
        if "src" in parts:
            src_index = parts.index("src")
            suffix = parts[src_index + 1 :]
            if suffix:
                return Path(*suffix).as_posix()
        return path.as_posix().lstrip("./")

    def _apply_class_nesting(
        self,
        tree: cst.Module,
        mappings: dict[str, str],
        changes: list[str],
    ) -> cst.Module:
        transformer = ClassNestingTransformer(mappings=mappings)
        updated_tree = tree.visit(transformer)
        if updated_tree.code != tree.code:
            changes.append(
                f"Applied ClassNestingTransformer ({len(mappings)} mappings)"
            )
        return updated_tree

    def _apply_helper_consolidation(
        self,
        tree: cst.Module,
        mappings: dict[str, str],
        changes: list[str],
    ) -> cst.Module:
        transformer = HelperConsolidationTransformer(helper_mappings=mappings)
        updated_tree = tree.visit(transformer)
        if updated_tree.code != tree.code:
            changes.append(
                f"Applied HelperConsolidationTransformer ({len(mappings)} mappings)"
            )
        return updated_tree

    def _apply_nested_class_propagation(
        self,
        tree: cst.Module,
        mappings: dict[str, str],
        changes: list[str],
    ) -> cst.Module:
        transformer = NestedClassPropagationTransformer(class_renames=mappings)
        wrapped_tree = MetadataWrapper(tree)
        updated_tree = wrapped_tree.visit(transformer)
        if updated_tree.code != tree.code:
            changes.append(
                f"Applied NestedClassPropagationTransformer ({len(mappings)} renames)"
            )
        return updated_tree


__all__ = ["ClassNestingRefactorRule", "RefactorResult"]
