"""Make command registry utilities for flext-tests."""

from __future__ import annotations

from pathlib import Path

from flext_tests import c, m, p, r, t
from flext_tests._utilities.make_contract import FlextTestsMakeContractUtilitiesMixin
from flext_tests._utilities.make_parsing import FlextTestsMakeParsingUtilitiesMixin


class FlextTestsMakeRegistryUtilitiesMixin(FlextTestsMakeContractUtilitiesMixin):
    """Build and resolve the Make command registry."""

    @staticmethod
    def _path_name(path: Path) -> str:
        """Return the stable basename used for registry discovery ordering."""
        return path.name

    @staticmethod
    def _make_add_command(
        commands_by_verb: t.MutableMappingKV[
            str, t.MutableMappingKV[str, m.Tests.MakeCommand]
        ],
        aliases_by_name: t.MutableMappingKV[str, str],
        command: m.Tests.MakeCommand,
    ) -> p.Result[bool]:
        """Add one command to mutable registry assembly state."""
        by_what = commands_by_verb.setdefault(command.verb, {})
        if command.what in by_what:
            return r[bool].fail(
                f"duplicate command: {command.verb} WHAT={command.what}"
            )
        by_what[command.what] = command
        for alias in command.aliases:
            previous = aliases_by_name.get(alias)
            if previous and previous != command.verb:
                return r[bool].fail(
                    f"duplicate alias: {alias} points to {previous} and {command.verb}"
                )
            aliases_by_name[alias] = command.verb
        return r[bool].ok(True)

    @staticmethod
    def _make_command_from_data(
        path: Path, data: t.Tests.MakeTomlTable, verb: str, what: str
    ) -> p.Result[m.Tests.MakeCommand]:
        """Build a command model from validated TOML header data."""
        values: t.MutableStrMapping = {}
        for field in ("domain", "summary", "description", "example"):
            value_result = FlextTestsMakeParsingUtilitiesMixin.make_require_string(
                data, field, path
            )
            if value_result.failure:
                return r[m.Tests.MakeCommand].fail(
                    value_result.error or f"{field} missing"
                )
            values[field] = value_result.value

        mutates_result = FlextTestsMakeParsingUtilitiesMixin.make_require_bool(
            data, "mutates", path
        )
        mutates_when_result = (
            FlextTestsMakeParsingUtilitiesMixin.make_parse_mutation_conditions(
                data.get("mutates_when"), path
            )
        )
        aliases_result = FlextTestsMakeParsingUtilitiesMixin.make_parse_aliases(
            data.get("aliases"), path
        )
        params_result = FlextTestsMakeParsingUtilitiesMixin.make_parse_params(
            data.get("params"), path
        )
        rules_result = FlextTestsMakeParsingUtilitiesMixin.make_parse_string_list(
            data, "rules", path
        )
        target_result = (
            FlextTestsMakeParsingUtilitiesMixin.make_require_optional_string(
                data, "target", path
            )
        )
        target_env_result = FlextTestsMakeParsingUtilitiesMixin.make_parse_string_map(
            data.get("target_env"), path
        )
        surface_result = (
            FlextTestsMakeParsingUtilitiesMixin.make_require_optional_string(
                data, "surface", path
            )
        )
        for result in (
            mutates_result,
            mutates_when_result,
            aliases_result,
            params_result,
            rules_result,
            target_result,
            target_env_result,
            surface_result,
        ):
            if result.failure:
                return r[m.Tests.MakeCommand].fail(
                    result.error or "command metadata invalid"
                )

        command = m.Tests.MakeCommand(
            verb=verb,
            what=what,
            domain=values["domain"],
            summary=values["summary"],
            description=values["description"],
            example=values["example"],
            path=path,
            mutates=mutates_result.value,
            mutates_when=mutates_when_result.value,
            aliases=aliases_result.value,
            params=params_result.value,
            rules=rules_result.value,
            target=target_result.value,
            target_env=target_env_result.value,
            surface=surface_result.value,
        )
        contract_result = (
            FlextTestsMakeContractUtilitiesMixin.make_validate_command_contract(command)
        )
        if contract_result.failure:
            return r[m.Tests.MakeCommand].fail(
                contract_result.error or "command contract invalid"
            )
        return r[m.Tests.MakeCommand].ok(command)

    @classmethod
    def make_discover(cls, scripts_dir: Path) -> p.Result[m.Tests.MakeRegistry]:
        """Discover and validate promoted commands under ``scripts/cmd``."""
        if not scripts_dir.exists():
            return r[m.Tests.MakeRegistry].fail("diretorio scripts/cmd missing")

        commands_by_verb: t.MutableMappingKV[
            str, t.MutableMappingKV[str, m.Tests.MakeCommand]
        ] = {}
        aliases_by_name: t.MutableMappingKV[str, str] = {}
        for verb_dir in sorted(scripts_dir.iterdir(), key=cls._path_name):
            if (
                not verb_dir.is_dir()
                or verb_dir.name in c.Tests.MAKE_IGNORED_COMMAND_DIRS
            ):
                continue
            for path in sorted(verb_dir.iterdir(), key=cls._path_name):
                if path.name == "__pycache__":
                    continue
                load_result = cls.make_load_command(path, verb_dir.name)
                if load_result.failure:
                    return r[m.Tests.MakeRegistry].fail(
                        load_result.error or "command load failed"
                    )
                add_result = cls._make_add_command(
                    commands_by_verb, aliases_by_name, load_result.value
                )
                if add_result.failure:
                    return r[m.Tests.MakeRegistry].fail(
                        add_result.error or "command registration failed"
                    )

        registry = m.Tests.MakeRegistry(
            commands_by_verb=commands_by_verb, aliases_by_name=aliases_by_name
        )
        validate_result = FlextTestsMakeContractUtilitiesMixin.make_validate_registry(
            registry
        )
        if validate_result.failure:
            return r[m.Tests.MakeRegistry].fail(
                validate_result.error or "registry validation failed"
            )
        return r[m.Tests.MakeRegistry].ok(registry)

    @classmethod
    def make_load_command(
        cls, path: Path, expected_verb: str
    ) -> p.Result[m.Tests.MakeCommand]:
        """Load one promoted command from its flext-command TOML header."""
        if path.name == "__pycache__":
            return r[m.Tests.MakeCommand].fail(f"{path}: cache publico invalido")
        if path.is_dir():
            return r[m.Tests.MakeCommand].fail(
                f"{path}: diretorio publico cannot estar aninhado"
            )
        if path.suffix not in c.Tests.MAKE_COMMAND_SUFFIXES:
            return r[m.Tests.MakeCommand].fail(
                f"{path}: file publico must be .sh ou .py"
            )
        data_result = FlextTestsMakeParsingUtilitiesMixin.make_header_data(path)
        if data_result.failure:
            return r[m.Tests.MakeCommand].fail(
                data_result.error or "header load failed"
            )
        data = data_result.value
        verb_result = FlextTestsMakeParsingUtilitiesMixin.make_require_string(
            data, "verb", path
        )
        what_result = FlextTestsMakeParsingUtilitiesMixin.make_require_string(
            data, "what", path
        )
        if verb_result.failure:
            return r[m.Tests.MakeCommand].fail(verb_result.error or "verb missing")
        if what_result.failure:
            return r[m.Tests.MakeCommand].fail(what_result.error or "what missing")
        verb = verb_result.value
        what = what_result.value
        if verb != expected_verb:
            return r[m.Tests.MakeCommand].fail(
                f"{path}: header verb={verb} diverge do diretorio {expected_verb}"
            )
        if what != path.stem:
            return r[m.Tests.MakeCommand].fail(
                f"{path}: header what={what} diverge do file {path.stem}"
            )
        return cls._make_command_from_data(path, data, verb, what)


__all__: list[str] = ["FlextTestsMakeRegistryUtilitiesMixin"]
