"""Make command parsing utilities for flext-tests."""

from __future__ import annotations

import ast
from pathlib import Path

from flext_cli import u
from flext_tests import c, m, p, r, t


class FlextTestsMakeParsingUtilitiesMixin:
    """Parse Make command definitions and parameters."""

    @staticmethod
    def make_header_data(path: Path) -> p.Result[t.Tests.MakeTomlTable]:
        """Return parsed TOML metadata from one command header."""
        try:
            lines = path.read_text(encoding="utf-8").splitlines()[:220]
        except OSError as exc:
            return r[t.Tests.MakeTomlTable].fail_op("command header read", exc)

        in_header = False
        payload: list[str] = []
        for raw in lines:
            stripped = raw.strip()
            content = stripped[1:].strip() if stripped.startswith("#") else stripped
            if content == c.Tests.MAKE_HEADER_START:
                in_header = True
                continue
            if in_header and content == c.Tests.MAKE_HEADER_END:
                break
            if in_header:
                payload.append(content)
        if not payload:
            return r[t.Tests.MakeTomlTable].fail(f"{path}: sem header flext-command")
        mapping = u.Cli.toml_mapping_from_text("\n".join(payload))
        if mapping is None:
            return r[t.Tests.MakeTomlTable].fail(f"{path}: header TOML invalido")
        try:
            table = t.Tests.MAKE_TOML_TABLE_ADAPTER.validate_python(mapping)
        except (TypeError, ValueError) as exc:
            return r[t.Tests.MakeTomlTable].fail(f"{path}: header TOML invalido: {exc}")
        return r[t.Tests.MakeTomlTable].ok(table)

    @staticmethod
    def make_has_executable_body(path: Path) -> p.Result[bool]:
        """Return whether a command file has executable Python outside metadata."""
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            return r[bool].fail_op("command body read", exc)

        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            return r[bool].fail(f"{path}: Python invalido: {exc}")

        body = tuple(tree.body)
        if not body:
            return r[bool].ok(False)
        first = body[0]
        has_module_docstring = (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        )
        return r[bool].ok(len(body) > 1 if has_module_docstring else True)

    @staticmethod
    def make_require_string(
        data: t.Tests.MakeTomlTable, key: str, path: Path
    ) -> p.Result[str]:
        """Return one required non-empty string field."""
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            return r[str].fail(f"{path}: required field missing: {key}")
        return r[str].ok(value.strip())

    @staticmethod
    def make_require_optional_string(
        data: t.Tests.MakeTomlTable, key: str, path: Path
    ) -> p.Result[str]:
        """Return one optional string field."""
        value = data.get(key, "")
        if not isinstance(value, str):
            return r[str].fail(f"{path}: optional field {key} must be a string")
        return r[str].ok(value.strip())

    @staticmethod
    def make_require_bool(
        data: t.Tests.MakeTomlTable, key: str, path: Path
    ) -> p.Result[bool]:
        """Return one required boolean field."""
        value = data.get(key)
        if not isinstance(value, bool):
            return r[bool].fail(f"{path}: required boolean field missing: {key}")
        return r[bool].ok(value)

    @staticmethod
    def make_parse_aliases(
        value: t.Tests.MakeTomlValue | None, path: Path
    ) -> p.Result[t.StrSequence]:
        """Parse optional command aliases."""
        return FlextTestsMakeParsingUtilitiesMixin.make_parse_string_value_list(
            value, "aliases", path
        )

    @staticmethod
    def make_parse_string_list(
        data: t.Tests.MakeTomlTable, field: str, path: Path
    ) -> p.Result[t.StrSequence]:
        """Parse one optional string-list field from a TOML table."""
        return FlextTestsMakeParsingUtilitiesMixin.make_parse_string_value_list(
            data.get(field), field, path
        )

    @staticmethod
    def make_parse_string_value_list(
        value: t.Tests.MakeTomlValue | None, field: str, path: Path
    ) -> p.Result[t.StrSequence]:
        """Parse one optional string-list value."""
        if value is None:
            return r[t.StrSequence].ok(())
        if not isinstance(value, list):
            return r[t.StrSequence].fail(f"{path}: {field} must be a list of strings")
        values: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                return r[t.StrSequence].fail(f"{path}: {field} invalido")
            values.append(item.strip())
        return r[t.StrSequence].ok(tuple(values))

    @staticmethod
    def make_parse_params(
        value: t.Tests.MakeTomlValue | None, path: Path
    ) -> p.Result[t.SequenceOf[m.Tests.MakeParam]]:
        """Parse command parameter declarations."""
        if value is None:
            return r[t.SequenceOf[m.Tests.MakeParam]].ok(())
        if not isinstance(value, list):
            return r[t.SequenceOf[m.Tests.MakeParam]].fail(
                f"{path}: params must contain a list"
            )
        params: list[m.Tests.MakeParam] = []
        for item in value:
            if not isinstance(item, dict):
                return r[t.SequenceOf[m.Tests.MakeParam]].fail(
                    f"{path}: params must contain TOML objects"
                )
            parsed = t.Tests.MAKE_TOML_TABLE_ADAPTER.validate_python(item)
            param_result = FlextTestsMakeParsingUtilitiesMixin.make_parse_param(
                parsed, path
            )
            if param_result.failure:
                return r[t.SequenceOf[m.Tests.MakeParam]].fail(
                    param_result.error or "param invalid"
                )
            params.append(param_result.value)
        return r[t.SequenceOf[m.Tests.MakeParam]].ok(tuple(params))

    @staticmethod
    def make_parse_mutation_conditions(
        value: t.Tests.MakeTomlValue | None, path: Path
    ) -> p.Result[t.SequenceOf[m.Tests.MakeMutationCondition]]:
        """Parse optional conditional mutation predicates."""
        if value is None:
            return r[t.SequenceOf[m.Tests.MakeMutationCondition]].ok(())
        if not isinstance(value, list):
            return r[t.SequenceOf[m.Tests.MakeMutationCondition]].fail(
                f"{path}: mutates_when must contain a list"
            )
        conditions: list[m.Tests.MakeMutationCondition] = []
        for item in value:
            if not isinstance(item, dict):
                return r[t.SequenceOf[m.Tests.MakeMutationCondition]].fail(
                    f"{path}: mutates_when must contain TOML objects"
                )
            parsed = t.Tests.MAKE_TOML_TABLE_ADAPTER.validate_python(item)
            condition_result = (
                FlextTestsMakeParsingUtilitiesMixin.make_parse_mutation_condition(
                    parsed, path
                )
            )
            if condition_result.failure:
                return r[t.SequenceOf[m.Tests.MakeMutationCondition]].fail(
                    condition_result.error or "mutation condition invalid"
                )
            conditions.append(condition_result.value)
        return r[t.SequenceOf[m.Tests.MakeMutationCondition]].ok(tuple(conditions))

    @staticmethod
    def make_parse_mutation_condition(
        data: t.Tests.MakeTomlTable, path: Path
    ) -> p.Result[m.Tests.MakeMutationCondition]:
        """Parse one conditional mutation predicate."""
        name_result = FlextTestsMakeParsingUtilitiesMixin.make_require_string(
            data, "name", path
        )
        values_result = FlextTestsMakeParsingUtilitiesMixin.make_parse_string_list(
            data, "values", path
        )
        if name_result.failure:
            return r[m.Tests.MakeMutationCondition].fail(
                name_result.error or "condition name missing"
            )
        if values_result.failure:
            return r[m.Tests.MakeMutationCondition].fail(
                values_result.error or "condition values invalid"
            )
        if not values_result.value:
            return r[m.Tests.MakeMutationCondition].fail(
                f"{path}: mutates_when.values cannot be empty"
            )
        return r[m.Tests.MakeMutationCondition].ok(
            m.Tests.MakeMutationCondition(
                name=name_result.value, values=values_result.value
            )
        )

    @staticmethod
    def make_parse_param(
        data: t.Tests.MakeTomlTable, path: Path
    ) -> p.Result[m.Tests.MakeParam]:
        """Parse one command parameter object."""
        name_result = FlextTestsMakeParsingUtilitiesMixin.make_require_string(
            data, "name", path
        )
        help_result = FlextTestsMakeParsingUtilitiesMixin.make_require_string(
            data, "help", path
        )
        if name_result.failure:
            return r[m.Tests.MakeParam].fail(name_result.error or "param name missing")
        if help_result.failure:
            return r[m.Tests.MakeParam].fail(help_result.error or "param help missing")
        required_raw = data.get("required", False)
        default_raw = data.get("default", "")
        if not isinstance(required_raw, bool):
            return r[m.Tests.MakeParam].fail(
                f"{path}: params.required must be a boolean"
            )
        if not isinstance(default_raw, str):
            return r[m.Tests.MakeParam].fail(f"{path}: params.default must be a string")
        choices_result = FlextTestsMakeParsingUtilitiesMixin.make_parse_string_list(
            data, "choices", path
        )
        if choices_result.failure:
            return r[m.Tests.MakeParam].fail(choices_result.error or "choices invalid")
        return r[m.Tests.MakeParam].ok(
            m.Tests.MakeParam(
                name=name_result.value,
                help=help_result.value,
                required=required_raw,
                default=default_raw,
                choices=choices_result.value,
            )
        )

    @staticmethod
    def make_parse_string_map(
        value: t.Tests.MakeTomlValue | None, path: Path
    ) -> p.Result[t.StrPairSequence]:
        """Parse an optional string-to-string mapping."""
        if value is None:
            return r[t.StrPairSequence].ok(())
        if not isinstance(value, dict):
            return r[t.StrPairSequence].fail(
                f"{path}: target_env must be a TOML object"
            )
        items: list[tuple[str, str]] = []
        for key, item in sorted(value.items()):
            if not key.strip():
                return r[t.StrPairSequence].fail(
                    f"{path}: target_env possui chave invalida"
                )
            if not isinstance(item, str):
                return r[t.StrPairSequence].fail(
                    f"{path}: target_env.{key} must be a string"
                )
            items.append((key.strip(), item))
        return r[t.StrPairSequence].ok(tuple(items))


__all__: list[str] = ["FlextTestsMakeParsingUtilitiesMixin"]
