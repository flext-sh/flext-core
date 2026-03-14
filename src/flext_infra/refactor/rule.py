"""Rule loader for flext_infra.refactor."""

from __future__ import annotations

import fnmatch
from collections.abc import Callable, Mapping
from pathlib import Path

from pydantic import JsonValue, TypeAdapter, ValidationError

from flext_core import r
from flext_infra import c, m, t, u
from flext_infra.refactor._base_rule import FlextInfraRefactorRule
from flext_infra.refactor.rules.class_nesting import ClassNestingRefactorRule
from flext_infra.refactor.validation import FlextInfraRefactorRuleDefinitionValidator


class FlextInfraRefactorRuleLoader:
    """Load and resolve refactor rules from YAML configuration files."""

    def __init__(self, config_path: Path) -> None:
        """Initialize with path to the refactor engine configuration file."""
        self.config_path = config_path

    def load_config(self) -> r[Mapping[str, JsonValue]]:
        """Load and validate the refactor engine configuration."""
        try:
            loaded = u.Infra.safe_load_yaml(self.config_path)
            normalized: dict[str, t.Infra.InfraValue] = TypeAdapter(
                dict[str, t.Infra.InfraValue]
            ).validate_python(dict(loaded.items()))
            scope_raw = normalized.get("refactor_engine")
            scope_map = self._normalize_str_object_mapping(scope_raw)
            scope = m.Infra.Refactor.EngineConfig.model_validate(scope_map)
            normalized["refactor_engine"] = scope.model_dump(mode="python")
            return r[Mapping[str, JsonValue]].ok(normalized)
        except (OSError, TypeError, ValueError) as exc:
            return r[Mapping[str, JsonValue]].fail(f"Failed to load config: {exc}")

    def extract_engine_file_filters(
        self,
        config: t.Infra.InfraValue,
    ) -> tuple[list[str], list[str]]:
        """Extract ignore patterns and file extensions from engine config."""
        scope = self._resolve_engine_config(config)
        return (list(scope.ignore_patterns), list(scope.file_extensions))

    def extract_project_scan_dirs(self, config: t.Infra.InfraValue) -> list[str]:
        """Extract project scan directories from engine config."""
        scope = self._resolve_engine_config(config)
        return list(scope.project_scan_dirs)

    def load_rules(
        self,
        rule_filters: list[str],
        validator: FlextInfraRefactorRuleDefinitionValidator,
        build_rule: Callable[
            [Mapping[str, t.Infra.InfraValue]],
            FlextInfraRefactorRule | None,
        ],
        build_file_rules: Callable[[], list[ClassNestingRefactorRule]],
    ) -> r[tuple[list[FlextInfraRefactorRule], list[ClassNestingRefactorRule]]]:
        """Load rules from YAML files, validate, and build rule instances."""
        try:
            rules_dir = self.config_path.parent / c.Infra.ReportKeys.RULES
            loaded_rules: list[FlextInfraRefactorRule] = []
            loaded_file_rules = build_file_rules()
            unknown_rules: list[str] = []
            for rule_file in sorted(rules_dir.glob("*.yml")):
                try:
                    rule_config: dict[str, t.Infra.InfraValue] = TypeAdapter(
                        dict[str, t.Infra.InfraValue]
                    ).validate_python(dict(u.Infra.safe_load_yaml(rule_file).items()))
                except (OSError, TypeError):
                    continue
                typed_rules = self._coerce_rule_definitions(
                    rule_config.get(c.Infra.ReportKeys.RULES),
                )
                for typed_rule_def in typed_rules:
                    if c.Infra.ReportKeys.ID not in typed_rule_def:
                        continue
                    if not typed_rule_def.get(c.Infra.ReportKeys.ENABLED, True):
                        continue
                    rule_id = str(typed_rule_def[c.Infra.ReportKeys.ID]).strip()
                    rule_id_lower = rule_id.lower()
                    matches_active_filters = not rule_filters or any(
                        fnmatch.fnmatch(rule_id_lower, active_filter.lower())
                        or active_filter.lower() in rule_id_lower
                        for active_filter in rule_filters
                    )
                    fix_action = (
                        str(
                            typed_rule_def.get(
                                c.Infra.ReportKeys.FIX_ACTION,
                                typed_rule_def.get(c.Infra.ReportKeys.ACTION, ""),
                            ),
                        )
                        .strip()
                        .lower()
                    )
                    if fix_action == "nest_classes":
                        continue
                    if not matches_active_filters:
                        continue
                    rule_validation = validator.validate_rule_definition(typed_rule_def)
                    if rule_validation is not None:
                        unknown_rules.append(rule_validation)
                        continue
                    rule = build_rule(typed_rule_def)
                    if rule is None:
                        unknown_rules.append(
                            str(
                                typed_rule_def.get(
                                    c.Infra.ReportKeys.ID,
                                    c.Infra.Defaults.UNKNOWN,
                                ),
                            ),
                        )
                        continue
                    loaded_rules.append(rule)
            if unknown_rules:
                unknown = ", ".join(sorted(unknown_rules))
                return r[
                    tuple[
                        list[FlextInfraRefactorRule],
                        list[ClassNestingRefactorRule],
                    ]
                ].fail(f"Unknown rule mapping for: {unknown}")
            return r[
                tuple[list[FlextInfraRefactorRule], list[ClassNestingRefactorRule]]
            ].ok((loaded_rules, loaded_file_rules))
        except Exception as exc:
            return r[
                tuple[list[FlextInfraRefactorRule], list[ClassNestingRefactorRule]]
            ].fail(f"Failed to load rules: {exc}")

    def _resolve_engine_config(
        self,
        config: t.Infra.InfraValue,
    ) -> m.Infra.Refactor.EngineConfig:
        config_map = self._normalize_str_object_mapping(config)
        scope_raw = config_map.get("refactor_engine")
        scope_map = self._normalize_str_object_mapping(scope_raw)
        return m.Infra.Refactor.EngineConfig.model_validate(scope_map)

    @staticmethod
    def _coerce_rule_definitions(
        value: t.Infra.InfraValue | None,
    ) -> list[dict[str, t.Infra.InfraValue]]:
        try:
            entries: list[JsonValue] = TypeAdapter(
                list[JsonValue],
            ).validate_python(value)
        except ValidationError:
            return []
        definitions: list[dict[str, t.Infra.InfraValue]] = []
        for item in entries:
            normalized = FlextInfraRefactorRuleLoader._normalize_str_object_mapping(
                item
            )
            if not normalized:
                continue
            definitions.append(normalized)
        return definitions

    @staticmethod
    def _normalize_str_object_mapping(
        value: t.Infra.InfraValue | None,
    ) -> dict[str, t.Infra.InfraValue]:
        try:
            adapter: TypeAdapter[dict[str, t.Infra.InfraValue]] = TypeAdapter(
                dict[str, t.Infra.InfraValue]
            )
            return adapter.validate_python(value)
        except ValidationError:
            return {}


__all__ = ["FlextInfraRefactorRule", "FlextInfraRefactorRuleLoader"]
