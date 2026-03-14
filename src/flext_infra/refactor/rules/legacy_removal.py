"""Rule that removes legacy compatibility code patterns."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import override

import libcst as cst
from pydantic import TypeAdapter, ValidationError

from flext_infra import c, t
from flext_infra.refactor.rule import FlextInfraRefactorRule
from flext_infra.refactor.transformers.alias_remover import (
    FlextInfraRefactorAliasRemover,
)
from flext_infra.refactor.transformers.deprecated_remover import (
    FlextInfraRefactorDeprecatedRemover,
)
from flext_infra.refactor.transformers.import_bypass_remover import (
    FlextInfraRefactorImportBypassRemover,
)


class FlextInfraRefactorLegacyRemovalRule(FlextInfraRefactorRule):
    """Remove aliases, deprecated classes, wrappers and import bypass blocks."""

    _CONFIG_ADAPTER: TypeAdapter[dict[str, t.Infra.InfraValue]] | None = None

    @staticmethod
    def _get_config_adapter() -> TypeAdapter[dict[str, t.Infra.InfraValue]]:
        """Get or create TypeAdapter for dict[str, t.Infra.InfraValue]."""
        if FlextInfraRefactorLegacyRemovalRule._CONFIG_ADAPTER is None:
            FlextInfraRefactorLegacyRemovalRule._CONFIG_ADAPTER = TypeAdapter(
                dict[str, t.Infra.InfraValue],
            )
        return FlextInfraRefactorLegacyRemovalRule._CONFIG_ADAPTER

    def _typed_config(self) -> dict[str, t.Infra.InfraValue]:
        """Get self.config validated as dict[str, InfraValue]."""
        return self._get_config_adapter().validate_python(self.config)

    @staticmethod
    def _is_forwarding_compatible(
        *,
        positional_expected: list[str],
        kwonly_expected: list[str],
        expected_star_arg: str | None,
        expected_star_kwarg: str | None,
        positional_forwarded: list[str],
        keyword_forwarded: Mapping[str, str],
        star_forwarded: str | None,
        star_kw_forwarded: str | None,
    ) -> bool:
        positional_by_name = len(positional_forwarded) == 0 and set(
            keyword_forwarded.keys(),
        ) >= set(positional_expected)
        positional_by_position = positional_forwarded == positional_expected
        if not (positional_by_position or positional_by_name):
            return False
        for name in positional_expected:
            if name in keyword_forwarded and keyword_forwarded[name] != name:
                return False
        if kwonly_expected and (
            not set(keyword_forwarded.keys()) >= set(kwonly_expected)
        ):
            return False
        for name in kwonly_expected:
            if keyword_forwarded.get(name) != name:
                return False
        extra_keywords = (
            set(keyword_forwarded.keys())
            - set(positional_expected)
            - set(kwonly_expected)
        )
        if extra_keywords:
            return False
        if expected_star_arg is not None and star_forwarded != expected_star_arg:
            return False
        if expected_star_arg is None and star_forwarded is not None:
            return False
        if expected_star_kwarg is not None and star_kw_forwarded != expected_star_kwarg:
            return False
        return not (expected_star_kwarg is None and star_kw_forwarded is not None)

    @staticmethod
    def _name_value(expr: cst.BaseExpression) -> str | None:
        value = getattr(expr, "value", None)
        if isinstance(value, str):
            return value
        return None

    @override
    def apply(
        self,
        tree: cst.Module,
        _file_path: Path | None = None,
    ) -> tuple[cst.Module, list[str]]:
        """Apply configured legacy-removal transforms to module tree."""
        changes: list[str] = []
        fix_action = (
            str(self.config.get(c.Infra.ReportKeys.FIX_ACTION, "")).strip().lower()
        )
        if "alias" in self.rule_id or fix_action == "remove":
            tree, alias_changes = self._remove_aliases(tree)
            changes.extend(alias_changes)
        if "deprecated" in self.rule_id or fix_action == "remove_and_update_refs":
            tree, deprecated_changes = self._remove_deprecated(tree)
            changes.extend(deprecated_changes)
        if "wrapper" in self.rule_id or fix_action == "inline_and_remove":
            tree, wrapper_changes = self._remove_wrappers(tree)
            changes.extend(wrapper_changes)
        if "bypass" in self.rule_id or fix_action == "keep_try_only":
            tree, bypass_changes = self._remove_import_bypasses(tree)
            changes.extend(bypass_changes)
        return (tree, changes)

    @staticmethod
    def _normalize_string_items(value: t.Infra.InfraValue) -> list[str]:
        if not isinstance(value, (list, tuple, set)):
            return []
        try:
            items: list[t.Infra.InfraValue] = TypeAdapter(
                list[t.Infra.InfraValue]
            ).validate_python(value)
        except ValidationError:
            return []
        output: list[str] = [item for item in items if isinstance(item, str)]
        return output

    def _expected_forwarding_params(
        self,
        func: cst.FunctionDef,
    ) -> tuple[list[str], list[str], str | None, str | None]:
        posonly_names = [param.name.value for param in func.params.posonly_params]
        positional_names = [param.name.value for param in func.params.params]
        positional_expected = posonly_names + positional_names
        kwonly_expected = [param.name.value for param in func.params.kwonly_params]
        expected_star_arg: str | None = None
        if isinstance(func.params.star_arg, cst.Param):
            expected_star_arg = func.params.star_arg.name.value
        expected_star_kwarg: str | None = None
        if isinstance(func.params.star_kwarg, cst.Param):
            expected_star_kwarg = func.params.star_kwarg.name.value
        return (
            positional_expected,
            kwonly_expected,
            expected_star_arg,
            expected_star_kwarg,
        )

    def _extract_passthrough_call(
        self,
        func: cst.FunctionDef,
    ) -> tuple[str, list[cst.Arg]] | None:
        if not isinstance(func.body, cst.IndentedBlock):
            return None
        if len(func.body.body) != 1:
            return None
        stmt = func.body.body[0]
        if not isinstance(stmt, cst.SimpleStatementLine):
            return None
        if len(stmt.body) != 1:
            return None
        small_stmt = stmt.body[0]
        if not isinstance(small_stmt, cst.Return):
            return None
        if not isinstance(small_stmt.value, cst.Call):
            return None
        if not isinstance(small_stmt.value.func, cst.Name):
            return None
        return (small_stmt.value.func.value, list(small_stmt.value.args))

    def _get_passthrough_target(self, func: cst.FunctionDef) -> str | None:
        passthrough = self._extract_passthrough_call(func)
        if passthrough is None:
            return None
        target_name, call_args = passthrough
        positional_expected, kwonly_expected, expected_star_arg, expected_star_kwarg = (
            self._expected_forwarding_params(func)
        )
        parsed_forwarding = self._parse_forwarded_arguments(call_args)
        if parsed_forwarding is None:
            return None
        positional_forwarded, keyword_forwarded, star_forwarded, star_kw_forwarded = (
            parsed_forwarding
        )
        if not self._is_forwarding_compatible(
            positional_expected=positional_expected,
            kwonly_expected=kwonly_expected,
            expected_star_arg=expected_star_arg,
            expected_star_kwarg=expected_star_kwarg,
            positional_forwarded=positional_forwarded,
            keyword_forwarded=keyword_forwarded,
            star_forwarded=star_forwarded,
            star_kw_forwarded=star_kw_forwarded,
        ):
            return None
        return target_name

    def _parse_forwarded_arguments(
        self,
        call_args: list[cst.Arg],
    ) -> tuple[list[str], dict[str, str], str | None, str | None] | None:
        positional_forwarded: list[str] = []
        keyword_forwarded: dict[str, str] = {}
        star_forwarded: str | None = None
        star_kw_forwarded: str | None = None
        for arg in call_args:
            if arg.keyword is not None:
                name_value = self._name_value(arg.value)
                if name_value is None:
                    return None
                keyword_forwarded[arg.keyword.value] = name_value
                continue
            if arg.star == "*":
                name_value = self._name_value(arg.value)
                if name_value is None:
                    return None
                star_forwarded = name_value
                continue
            if arg.star == "**":
                name_value = self._name_value(arg.value)
                if name_value is None:
                    return None
                star_kw_forwarded = name_value
                continue
            if arg.star not in {"", None}:
                return None
            name_value = self._name_value(arg.value)
            if name_value is None:
                return None
            positional_forwarded.append(name_value)
        return (
            positional_forwarded,
            keyword_forwarded,
            star_forwarded,
            star_kw_forwarded,
        )

    def _remove_aliases(self, tree: cst.Module) -> tuple[cst.Module, list[str]]:
        typed_config = self._typed_config()
        allow_aliases = set(
            self._normalize_string_items(typed_config.get("allow_aliases", []))
        )
        allow_target_suffixes = tuple(
            self._normalize_string_items(typed_config.get("allow_target_suffixes", [])),
        )
        transformer = FlextInfraRefactorAliasRemover(
            allow_aliases=allow_aliases,
            allow_target_suffixes=allow_target_suffixes,
        )
        new_tree = tree.visit(transformer)
        return (new_tree, transformer.changes)

    def _remove_deprecated(self, tree: cst.Module) -> tuple[cst.Module, list[str]]:
        transformer = FlextInfraRefactorDeprecatedRemover()
        new_tree = tree.visit(transformer)
        return (new_tree, transformer.changes)

    def _remove_import_bypasses(self, tree: cst.Module) -> tuple[cst.Module, list[str]]:
        transformer = FlextInfraRefactorImportBypassRemover()
        new_tree = tree.visit(transformer)
        return (new_tree, transformer.changes)

    def _remove_wrappers(self, tree: cst.Module) -> tuple[cst.Module, list[str]]:
        changes: list[str] = []
        new_body: list[cst.BaseStatement] = []
        for stmt in tree.body:
            if not isinstance(stmt, cst.FunctionDef):
                new_body.append(stmt)
                continue
            target_name = self._get_passthrough_target(stmt)
            if target_name is None:
                new_body.append(stmt)
                continue
            alias_assign = cst.SimpleStatementLine(
                body=[
                    cst.Assign(
                        targets=[cst.AssignTarget(target=cst.Name(stmt.name.value))],
                        value=cst.Name(target_name),
                    ),
                ],
            )
            new_body.append(alias_assign)
            changes.append(f"Inlined wrapper: {stmt.name.value} -> {target_name}")
        return (tree.with_changes(body=new_body), changes)


__all__ = ["FlextInfraRefactorLegacyRemovalRule"]
