"""Direct text-rule execution for the refactor service."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra import c, m, t, u
from flext_infra.refactor.legacy_text_ops import FlextInfraRefactorLegacyTextOps
from flext_infra.transformers.class_reconstructor import (
    FlextInfraRefactorClassReconstructor,
)
from flext_infra.transformers.future_import import FlextInfraRefactorFutureImport
from flext_infra.transformers.import_modernizer import (
    FlextInfraRefactorImportModernizer,
)
from flext_infra.transformers.lazy_import_fixer import FlextInfraRefactorLazyImportFixer
from flext_infra.transformers.signature_propagator import (
    FlextInfraRefactorSignaturePropagator,
)
from flext_infra.transformers.symbol_propagator import (
    FlextInfraRefactorSymbolPropagator,
)
from flext_infra.transformers.tier0_import_fixer import (
    FlextInfraTransformerTier0ImportFixer,
)
from flext_infra.transformers.typing_unifier import FlextInfraRefactorTypingUnifier

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p


class FlextInfraRefactorTextExecutor(FlextInfraRefactorLegacyTextOps):
    """Execute declarative text rules directly from kind + settings."""

    def _apply_text_rule_selection(
        self,
        kind: c.Infra.RefactorRuleKind,
        settings: t.MappingKV[str, t.Infra.InfraValue],
        source: str,
        file_path: Path,
    ) -> t.Infra.TransformResult:
        """Apply text rule selection."""
        match kind:
            case c.Infra.RefactorRuleKind.FUTURE_ANNOTATIONS:
                result = self._apply_future_annotations(source)
            case c.Infra.RefactorRuleKind.LEGACY_REMOVAL:
                result = self._apply_legacy_removal(settings, source)
            case c.Infra.RefactorRuleKind.IMPORT_MODERNIZER:
                result = self._apply_import_modernizer(settings, source)
            case c.Infra.RefactorRuleKind.CLASS_RECONSTRUCTOR:
                result = self._apply_class_reconstructor(settings, source)
            case c.Infra.RefactorRuleKind.PATTERN_CORRECTIONS:
                result = self._apply_pattern_corrections(settings, source, file_path)
            case c.Infra.RefactorRuleKind.TYPING_UNIFICATION:
                result = self._apply_typing_unification(source, file_path)
            case c.Infra.RefactorRuleKind.TYPING_ANNOTATION_FIX:
                result = self._apply_typing_annotation_fix(settings, source, file_path)
            case c.Infra.RefactorRuleKind.TIER0_IMPORT_FIX:
                result = self._apply_tier0_import_fix(settings, source, file_path)
            case c.Infra.RefactorRuleKind.SYMBOL_PROPAGATION:
                result = self._apply_symbol_propagation(settings, source)
            case c.Infra.RefactorRuleKind.SIGNATURE_PROPAGATION:
                result = self._apply_signature_propagation(settings, source)
        return result

    @staticmethod
    def _apply_change_tracker_transformer(
        transformer: p.Infra.ChangeTracker, source: str
    ) -> t.Infra.TransformResult:
        """Apply change tracker transformer."""
        apply_fn = getattr(transformer, "apply_to_source", None)
        if callable(apply_fn):
            result = apply_fn(source)
            new_source = (
                result[0]
                if isinstance(result, tuple) and result and isinstance(result[0], str)
                else source
            )
            return (new_source, list(getattr(transformer, "changes", list[str]())))
        return (source, list[str]())

    @staticmethod
    def _apply_future_annotations(source: str) -> t.Infra.TransformResult:
        """Apply future annotations."""
        return FlextInfraRefactorFutureImport().apply_to_source(source)

    def _apply_import_modernizer(
        self, settings: t.MappingKV[str, t.Infra.InfraValue], source: str
    ) -> t.Infra.TransformResult:
        """Apply import modernizer."""
        settings_mapping = t.Cli.JSON_MAPPING_ADAPTER.validate_python(settings)
        fix_action = u.Cli.json_get_str_key(
            settings_mapping, c.Infra.RK_FIX_ACTION, case="lower"
        )
        if fix_action == "hoist_to_module_top":
            return self._apply_change_tracker_transformer(
                FlextInfraRefactorLazyImportFixer(), source
            )
        runtime_aliases = set(u.runtime_alias_names(c.Infra.PKG_INFRA_UNDERSCORE))
        blocked = set(u.Infra.collect_blocked_aliases(source, runtime_aliases))
        blocked.update(u.Infra.collect_shadowed_aliases(source, runtime_aliases))
        forbidden = settings.get(c.Infra.RK_FORBIDDEN_IMPORTS)
        if forbidden is None:
            forbidden = t.Cli.JSON_LIST_ADAPTER.validate_python([
                t.Cli.JSON_MAPPING_ADAPTER.validate_python(settings)
            ])
        parsed_rules = tuple(u.Infra.parse_forbidden_rules(forbidden))
        modernizer = FlextInfraRefactorImportModernizer(
            imports_to_remove=[
                mod
                for rule_config in parsed_rules
                for mod in (
                    [rule_config.module]
                    + (
                        [rule_config.module.split(".", 1)[0]]
                        if "." in rule_config.module
                        else []
                    )
                )
            ],
            symbols_to_replace={
                key: value
                for rule_config in parsed_rules
                for key, value in rule_config.symbol_mapping.items()
            },
            runtime_aliases=runtime_aliases,
            blocked_aliases=blocked,
        )
        return modernizer.apply_to_source(source)

    def _apply_pattern_corrections(
        self,
        settings: t.MappingKV[str, t.Infra.InfraValue],
        source: str,
        file_path: Path,
    ) -> t.Infra.TransformResult:
        """Apply pattern corrections."""
        settings_mapping = t.Cli.JSON_MAPPING_ADAPTER.validate_python(settings)
        fix_action = u.Cli.json_get_str_key(
            settings_mapping, c.Infra.RK_FIX_ACTION, case="lower"
        )
        if fix_action == "fix_silent_failure_sentinels":
            sentinels_result: t.Infra.TransformResult = (
                u.Infra.apply_transformer_to_source(
                    source,
                    file_path,
                    lambda rope_project, resource: u.Infra.fix_silent_failure_sentinels(
                        rope_project, resource, apply=True
                    ),
                )
            )
            return sentinels_result
        if fix_action == "remove_redundant_casts":
            casts_result: t.Infra.TransformResult = u.Infra.apply_transformer_to_source(
                source, file_path, self._remove_redundant_casts
            )
            return casts_result
        mappings_result: t.Infra.TransformResult = u.Infra.apply_transformer_to_source(
            source, file_path, self._replace_mapping_annotations
        )
        return mappings_result

    def _apply_typing_unification(
        self, source: str, file_path: Path
    ) -> t.Infra.TransformResult:
        """Apply typing unification."""
        return self._apply_change_tracker_transformer(
            FlextInfraRefactorTypingUnifier(
                canonical_map=c.Infra.TYPING_INLINE_UNION_CANONICAL_MAP,
                file_path=file_path,
            ),
            source,
        )

    def _apply_typing_annotation_fix(
        self,
        settings: t.MappingKV[str, t.Infra.InfraValue],
        source: str,
        file_path: Path,
    ) -> t.Infra.TransformResult:
        """Apply typing annotation fix."""
        settings_mapping = t.Cli.JSON_MAPPING_ADAPTER.validate_python(settings)
        fix_action = u.Cli.json_get_str_key(
            settings_mapping, c.Infra.RK_FIX_ACTION, case="lower"
        )
        if fix_action != "replace_object_annotations":
            return (source, list[str]())
        result: t.Infra.TransformResult = u.Infra.apply_transformer_to_source(
            source, file_path, self._replace_object_annotations
        )
        return result

    @staticmethod
    def _replace_object_annotations(
        rope_project: t.Infra.RopeProject, resource: t.Infra.RopeResource
    ) -> t.Infra.TransformResult:
        """Replace object annotations."""
        updated_source, count = u.Infra.batch_replace_annotations(
            rope_project, resource, {"t.JsonValue": "t.JsonValue"}, apply=True
        )
        if count == 0:
            return (updated_source, list[str]())
        return (updated_source, ["Replaced annotation: t.JsonValue -> t.JsonValue"])

    @staticmethod
    def _remove_redundant_casts(
        rope_project: t.Infra.RopeProject, resource: t.Infra.RopeResource
    ) -> t.Infra.TransformResult:
        """Remove redundant casts."""
        updated_source, count = u.Infra.remove_redundant_cast(
            rope_project, resource, apply=True
        )
        if count == 0:
            return (updated_source, list[str]())
        return (updated_source, [f"Removed {count} redundant cast() calls"])

    @staticmethod
    def _replace_mapping_annotations(
        rope_project: t.Infra.RopeProject, resource: t.Infra.RopeResource
    ) -> t.Infra.TransformResult:
        """Replace mapping annotations."""
        _ = rope_project
        source = resource.read()
        total = 0
        source, alias_count = c.Infra.DICT_STR_JSONVALUE_RE.subn(
            "t.JsonMapping", source
        )
        total += alias_count
        source, generic_count = c.Infra.DICT_GENERIC_RE.subn("t.MappingKV[", source)
        total += generic_count
        if total > 0 and source != resource.read():
            resource.write(source)
        if total == 0:
            return (source, list[str]())
        return (
            source,
            [f"Converted {total} dict annotations to Mapping/t.JsonMapping"],
        )

    def _apply_tier0_import_fix(
        self,
        settings: t.MappingKV[str, t.Infra.InfraValue],
        source: str,
        file_path: Path,
    ) -> t.Infra.TransformResult:
        """Apply tier0 import fix."""
        analyzer = FlextInfraTransformerTier0ImportFixer.Analyzer(
            file_path=file_path,
            tier0_modules=self._tuple_setting(
                settings,
                c.Infra.RK_TIER0_MODULES,
                (c.Infra.CONSTANTS_PY, c.Infra.TYPINGS_PY, c.Infra.PROTOCOLS_PY),
            ),
            core_aliases=self._tuple_setting(
                settings,
                c.Infra.RK_CORE_ALIASES,
                tuple(
                    alias_name
                    for alias_name, module_name, _ in u.lazy_alias_suffixes(
                        c.Infra.PKG_INFRA_UNDERSCORE
                    )
                    if module_name.split(".", 1)[0] != c.Infra.PKG_INFRA_UNDERSCORE
                ),
            ),
        )
        analysis = analyzer.build_analysis()
        has_violations: bool = analysis.has_violations
        if not has_violations:
            return (source, list[str]())
        transformer = FlextInfraTransformerTier0ImportFixer.Transformer(
            analysis=analysis,
            alias_to_submodule=self._mapping_setting(
                settings, c.Infra.RK_ALIAS_TO_SUBMODULE
            ),
            core_package=str(
                settings.get(c.Infra.RK_CORE_PACKAGE, c.Infra.PKG_CORE_UNDERSCORE)
            ),
        )
        result: t.Infra.TransformResult = u.Infra.apply_transformer_to_source(
            source, file_path, transformer.transform
        )
        return result

    def _apply_symbol_propagation(
        self, settings: t.MappingKV[str, t.Infra.InfraValue], source: str
    ) -> t.Infra.TransformResult:
        """Apply symbol propagation."""
        transformer = FlextInfraRefactorSymbolPropagator(
            target_modules=set(
                u.Infra.string_list(settings.get(c.Infra.RK_TARGET_MODULES, []))
            ),
            module_renames=self._mapping_setting(settings, c.Infra.RK_MODULE_RENAMES),
            import_symbol_renames=self._mapping_setting(
                settings, c.Infra.RK_IMPORT_SYMBOL_RENAMES
            ),
        )
        return self._apply_change_tracker_transformer(transformer, source)

    def _apply_signature_propagation(
        self, settings: t.MappingKV[str, t.Infra.InfraValue], source: str
    ) -> t.Infra.TransformResult:
        """Apply signature propagation."""
        try:
            typed_items = t.Infra.CONTAINER_DICT_SEQ_ADAPTER.validate_python(
                settings.get(c.Infra.RK_SIGNATURE_MIGRATIONS, [])
            )
        except c.ValidationError:
            return (source, list[str]())
        parsed = [
            m.Infra.SignatureMigration.model_validate(item) for item in typed_items
        ]
        migrations: t.SequenceOf[m.Infra.SignatureMigration] = [
            item for item in parsed if item.enabled
        ]
        if not migrations:
            return (source, list[str]())
        transformer = FlextInfraRefactorSignaturePropagator(migrations=migrations)
        return (transformer.apply_to_source(source), list(transformer.changes))

    def _apply_class_reconstructor(
        self, settings: t.MappingKV[str, t.Infra.InfraValue], source: str
    ) -> t.Infra.TransformResult:
        """Apply class reconstructor."""
        try:
            order_config = t.Infra.CONTAINER_DICT_SEQ_ADAPTER.validate_python(
                settings.get(c.Infra.RK_METHOD_ORDER)
                or settings.get(c.Infra.RK_ORDER, [])
            )
        except c.ValidationError:
            return (source, list[str]())
        if not order_config:
            return (source, list[str]())
        return self._apply_change_tracker_transformer(
            FlextInfraRefactorClassReconstructor(order_config=order_config), source
        )

    @staticmethod
    def _tuple_setting(
        settings: t.MappingKV[str, t.Infra.InfraValue], key: str, default: t.StrTuple
    ) -> t.StrTuple:
        """Return tuple-valued setting."""
        value = settings.get(key, list(default))
        return (
            tuple(str(item) for item in value)
            if isinstance(value, list)
            else tuple(default)
        )

    @staticmethod
    def _mapping_setting(
        settings: t.MappingKV[str, t.Infra.InfraValue], key: str
    ) -> t.StrMapping:
        """Return a normalized string mapping setting."""
        mapping_value = u.Cli.json_as_mapping(settings.get(key, {}))
        normalized_mapping: dict[str, str] = {
            item_key: str(item_value) for item_key, item_value in mapping_value.items()
        }
        return normalized_mapping


__all__: list[str] = ["FlextInfraRefactorTextExecutor"]
