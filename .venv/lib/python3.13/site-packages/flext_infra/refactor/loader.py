"""Declarative rule loader for flext-infra refactor workflows."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from flext_cli import u
from flext_infra import c, t

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraRefactorRuleLoader:
    """Load refactor config and matched rule selections from declarative assets."""

    def __init__(self, config_path: Path) -> None:
        """Initialize loader state with one config path."""
        self.config_path = config_path
        # NOTE (multi-agent): orchestrator dispatch/scope mixins pass this
        # mapping to u.Infra.collect_refactor_*_files; it is populated by
        # load_config() and must stay on the loader (commit 0d1e1b7d dropped
        # it and broke those call sites).
        self.settings: t.MappingKV[str, t.Infra.InfraValue] = {}
        self.rules: t.MutableSequenceOf[
            t.Infra.RuleSelection[c.Infra.RefactorRuleKind]
        ] = []
        self.file_rules: t.MutableSequenceOf[
            t.Infra.RuleSelection[c.Infra.RefactorFileRuleKind]
        ] = []
        self.rule_filters: t.MutableSequenceOf[str] = []

    def load_config(self) -> p.Result[t.MappingKV[str, t.Infra.InfraValue]]:
        """Load YAML configuration for this refactor session."""
        result = u.Cli.rules_load_scoped_config(
            self.config_path,
            scope_key=c.Infra.RK_REFACTOR,
            allowed_keys=c.Infra.REFACTOR_CONFIG_KEYS,
        )
        if result.success:
            self.settings = t.Infra.INFRA_MAPPING_ADAPTER.validate_python(
                dict(result.value)
            )
            u.Cli.info(f"Loaded settings from {self.config_path}")
        return result

    def load_rules(
        self,
    ) -> p.Result[
        t.Infra.LoadedRuleSelections[
            c.Infra.RefactorRuleKind, c.Infra.RefactorFileRuleKind
        ]
    ]:
        """Load enabled text/file rule selections from declarative YAML assets."""
        result: p.Result[
            t.Infra.LoadedRuleSelections[
                c.Infra.RefactorRuleKind, c.Infra.RefactorFileRuleKind
            ]
        ] = u.Cli.rules_load_local_definitions(
            self.config_path,
            package_rules_dir=Path(__file__).resolve().parent.parent / c.Infra.RK_RULES,
            rule_filters=self.rule_filters,
            rule_catalog=c.Infra.RULE_MATCHERS_BY_KIND,
            file_rule_catalog=c.Infra.FILE_RULE_MATCHERS_BY_KIND,
            rules_key=c.Infra.RK_RULES,
            rule_id_key=c.Infra.RK_ID,
            enabled_key=c.Infra.RK_ENABLED,
        )
        if result.success:
            loaded_rules, loaded_file_rules = result.value
            self.rules = list(loaded_rules)
            self.file_rules = list(loaded_file_rules)
            u.Cli.info(f"Loaded {len(self.rules)} rules")
            if self.file_rules:
                u.Cli.info(f"Loaded {len(self.file_rules)} file rules")
            if self.rule_filters:
                u.Cli.info(f"Active filters: {', '.join(self.rule_filters)}")
        return result

    def set_rule_filters(self, filters: t.StrSequence) -> None:
        """Normalize and store active rule filters."""
        self.rule_filters = [item.lower() for item in filters]

    def list_rules(self) -> t.SequenceOf[t.FeatureFlagMapping]:
        """Return loaded rules metadata for listing and diagnostics."""
        return [
            {
                c.Infra.RK_ID: str(
                    settings.get(c.Infra.RK_ID, c.Infra.DEFAULT_UNKNOWN)
                ),
                c.NAME: str(
                    settings.get(
                        c.NAME, settings.get(c.Infra.RK_ID, c.Infra.DEFAULT_UNKNOWN)
                    )
                ),
                c.Infra.RK_DESCRIPTION: str(settings.get(c.Infra.RK_DESCRIPTION, "")),
                c.Infra.RK_ENABLED: bool(settings.get(c.Infra.RK_ENABLED, True)),
                c.Infra.RK_SEVERITY: str(
                    settings.get(c.Infra.RK_SEVERITY, c.Infra.SeverityLevel.WARNING)
                ),
            }
            for _, settings in self.rules
        ]

    @staticmethod
    def print_rules_table(rows: t.SequenceOf[t.FeatureFlagMapping]) -> None:
        """Render and print one rules table using the shared CLI table helpers."""
        data_result = u.Cli.tables_normalize_data(rows)
        if data_result.failure:
            u.Cli.error(data_result.error or "failed to normalize rules table")
            return
        settings_result = u.Cli.tables_resolve_config(
            headers=list(c.Infra.RULE_TABLE_HEADERS)
        )
        if settings_result.failure:
            u.Cli.error(settings_result.error or "failed to resolve table settings")
            return
        rendered_result = u.Cli.tables_render(data_result.value, settings_result.value)
        if rendered_result.failure:
            u.Cli.error(rendered_result.error or "failed to render rules table")
            return
        u.Cli.emit_raw(rendered_result.value)


__all__: list[str] = ["FlextInfraRefactorRuleLoader"]
