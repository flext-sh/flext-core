"""Rules Pydantic domain models for local rule loading."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from flext_cli import c, t
from flext_core import m


class FlextCliModelsRules:
    """Rules models namespace exposed through ``m.Cli``."""

    class LocalDefinitionsOptions[TRuleKind, TFileRuleKind](m.ArbitraryTypesModel):
        """Validated options envelope for loading local rule definitions."""

        package_rules_dir: Annotated[
            Path,
            m.Field(description="Directory containing packaged default rule YAMLs"),
        ]
        rule_filters: Annotated[
            t.StrSequence,
            m.Field(description="Glob/substring filters limiting loaded rules"),
        ]
        rule_catalog: Annotated[
            t.Cli.RuleCatalog[TRuleKind],
            m.Field(description="Catalog mapping rule kinds to their matchers"),
        ]
        file_rule_catalog: Annotated[
            t.Cli.RuleCatalog[TFileRuleKind] | None,
            m.Field(description="Optional catalog of file-rule kinds and matchers"),
        ] = None
        registry_filename: Annotated[
            str,
            m.Field(description="Name of the YAML registry file inside the rules dir"),
        ] = c.Cli.RULES_REGISTRY_FILENAME
        rules_key: Annotated[
            str,
            m.Field(description="Top-level key under which rules are nested in YAMLs"),
        ] = c.Cli.DICT_KEY_RULES
        rule_id_key: Annotated[
            str, m.Field(description="Per-rule key holding the unique rule identifier")
        ] = c.Cli.DICT_KEY_RULE_ID
        enabled_key: Annotated[
            str,
            m.Field(description="Per-rule key controlling whether the rule is active"),
        ] = c.Cli.DICT_KEY_ENABLED
        action_key: Annotated[
            str, m.Field(description="Per-rule key naming the fix action to dispatch")
        ] = c.Cli.RULES_ACTION_KEY
        fallback_action_key: Annotated[
            str,
            m.Field(description="Per-rule fallback key when ``action_key`` is absent"),
        ] = c.Cli.DICT_KEY_ACTION
        check_key: Annotated[
            str,
            m.Field(description="Per-rule key naming the check predicate to dispatch"),
        ] = c.Cli.DICT_KEY_CHECK
        rules_dir_name: Annotated[
            str,
            m.Field(description="Name of the rules subdirectory inside config_path"),
        ] = c.Cli.RULES_DIR_NAME


__all__: t.MutableSequenceOf[str] = ["FlextCliModelsRules"]
