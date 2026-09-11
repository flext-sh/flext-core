"""Make command contract utilities for flext-tests."""

from __future__ import annotations

from flext_tests import c, m, p, r, t
from flext_tests._utilities.make_parsing import FlextTestsMakeParsingUtilitiesMixin


class FlextTestsMakeContractUtilitiesMixin(FlextTestsMakeParsingUtilitiesMixin):
    """Validate Make command contracts and registries."""

    @staticmethod
    def make_param_value(
        param: m.Tests.MakeParam,
        command: m.Tests.MakeCommand,
        env: t.MappingKV[str, str],
    ) -> str:
        """Return the current environment-backed value for one parameter."""
        if param.name == c.Tests.MAKE_WHAT_PARAM:
            what: str = command.what
            return what
        name: str = param.name
        default: str = param.default
        value: str = env.get(name, default)
        return value.strip()

    @staticmethod
    def make_validate_invocation(
        command: m.Tests.MakeCommand,
        env: t.MappingKV[str, str],
        *,
        require_required: bool = True,
    ) -> p.Result[bool]:
        """Validate parameter values for one command invocation."""
        for param in command.params:
            value = FlextTestsMakeContractUtilitiesMixin.make_param_value(
                param, command, env
            )
            if require_required and param.required and not value:
                return r[bool].fail(
                    f"{command.verb} WHAT={command.what}: required parameter "
                    f"missing: {param.name}; exemplo: {command.example}"
                )
            if value and param.choices and value not in param.choices:
                valid = "|".join(param.choices)
                return r[bool].fail(
                    f"{command.verb} WHAT={command.what}: {param.name}={value!r} "
                    f"invalido; validos: {valid}"
                )
        return r[bool].ok(True)

    @staticmethod
    def make_validate_command_contract(command: m.Tests.MakeCommand) -> p.Result[bool]:
        """Validate one command against the generic dispatcher contract."""
        param_by_name = {param.name: param for param in command.params}
        if command.mutates:
            for name in c.Tests.MAKE_MUTATION_REQUIRED_PARAMS:
                param = param_by_name.get(name)
                if param is None or not param.required:
                    return r[bool].fail(
                        f"{command.path}: parameter {name} must be required"
                    )
            apply_param = param_by_name.get(c.Tests.MAKE_APPLY_PARAM)
            if (
                apply_param
                and c.Tests.MAKE_DISPATCH_ENV_VALUE not in apply_param.choices
            ):
                return r[bool].fail(
                    f"{command.path}: APPLY must declare choices containing Y"
                )
        if command.path.suffix not in c.Tests.MAKE_COMMAND_SUFFIXES:
            return r[bool].fail(
                f"{command.path}: command must use a .py or .sh extension"
            )
        if command.path.name != f"{command.what}.py" and command.path.suffix != ".sh":
            return r[bool].fail(
                f"{command.path}: command must use a .py or .sh extension"
            )
        if not command.summary.strip():
            return r[bool].fail(f"{command.path}: campo summary cannot be empty")
        if command.target and command.path.suffix != ".py":
            return r[bool].fail(
                f"{command.path}: header-only target must use a .py file"
            )
        if command.target_env and not command.target:
            return r[bool].fail(f"{command.path}: target_env exige target")
        if command.target:
            body_result = FlextTestsMakeParsingUtilitiesMixin.make_has_executable_body(
                command.path
            )
            if body_result.failure:
                return r[bool].fail(body_result.error or "target body check failed")
            if body_result.value:
                return r[bool].fail(
                    f"{command.path}: commands with a target must be header-only"
                )
        condition_result = (
            FlextTestsMakeContractUtilitiesMixin.make_validate_mutation_conditions(
                command, param_by_name
            )
        )
        if condition_result.failure:
            return r[bool].fail(condition_result.error or "mutates_when invalid")
        return r[bool].ok(True)

    @staticmethod
    def make_validate_mutation_conditions(
        command: m.Tests.MakeCommand, param_by_name: t.MappingKV[str, m.Tests.MakeParam]
    ) -> p.Result[bool]:
        """Validate conditional mutation predicates against declared parameters."""
        if not command.mutates_when:
            return r[bool].ok(True)
        apply_param = param_by_name.get(c.Tests.MAKE_APPLY_PARAM)
        if (
            apply_param is None
            or c.Tests.MAKE_DISPATCH_ENV_VALUE not in apply_param.choices
        ):
            return r[bool].fail(
                f"{command.path}: mutates_when exige APPLY com choice Y"
            )
        for condition in command.mutates_when:
            param = param_by_name.get(condition.name)
            if param is None:
                return r[bool].fail(
                    f"{command.path}: mutates_when references a missing parameter "
                    f"{condition.name}"
                )
            if param.choices:
                missing = tuple(
                    value for value in condition.values if value not in param.choices
                )
                if missing:
                    return r[bool].fail(
                        f"{command.path}: mutates_when.{condition.name} possui "
                        f"valores fora de choices: {','.join(missing)}"
                    )
        return r[bool].ok(True)

    @staticmethod
    def make_validate_registry(registry: m.Tests.MakeRegistry) -> p.Result[bool]:
        """Validate the complete discovered command registry."""
        if not registry.commands_by_verb:
            return r[bool].fail("no command found in scripts/cmd/<verb>/<what>")
        for verb, commands in sorted(registry.commands_by_verb.items()):
            if c.Tests.MAKE_DEFAULT_COMMAND not in commands:
                return r[bool].fail(
                    f"verb '{verb}' missing WHAT={c.Tests.MAKE_DEFAULT_COMMAND}"
                )
            domains = {command.domain for command in commands.values()}
            if len(domains) != 1:
                valid = ", ".join(sorted(domains))
                return r[bool].fail(
                    f"verb '{verb}' declares more than one domain: {valid}"
                )
            for command in commands.values():
                command_result = FlextTestsMakeContractUtilitiesMixin.make_validate_registered_command(
                    command
                )
                if command_result.failure:
                    return r[bool].fail(
                        command_result.error or "command contract invalid"
                    )
            choices_result = (
                FlextTestsMakeContractUtilitiesMixin.make_validate_all_choices(
                    verb, commands
                )
            )
            if choices_result.failure:
                return r[bool].fail(choices_result.error or "choices invalid")
        return r[bool].ok(True)

    @staticmethod
    def make_validate_registered_command(
        command: m.Tests.MakeCommand,
    ) -> p.Result[bool]:
        """Validate one command inside a complete registry."""
        if command.what != c.Tests.MAKE_DEFAULT_COMMAND and command.aliases:
            return r[bool].fail(
                f"{command.path}: aliases podem ser declarados apenas em "
                f"WHAT={c.Tests.MAKE_DEFAULT_COMMAND}"
            )
        return FlextTestsMakeContractUtilitiesMixin.make_validate_command_contract(
            command
        )

    @staticmethod
    def make_validate_all_choices(
        verb: str, commands: t.MappingKV[str, m.Tests.MakeCommand]
    ) -> p.Result[bool]:
        """Validate WHAT choices declared by the verb default command."""
        all_command = commands[c.Tests.MAKE_DEFAULT_COMMAND]
        what_param = next(
            (
                param
                for param in all_command.params
                if param.name == c.Tests.MAKE_WHAT_PARAM
            ),
            None,
        )
        if what_param is None or not what_param.choices:
            return r[bool].ok(True)
        declared = tuple(sorted(what_param.choices))
        actual = tuple(sorted(commands))
        if declared != actual:
            return r[bool].fail(
                f"{all_command.path}: WHAT choices diverge from the promoted commands "
                f"para {verb}: declared={','.join(declared)} actual={','.join(actual)}"
            )
        return r[bool].ok(True)

    @staticmethod
    def make_registry_verbs(registry: m.Tests.MakeRegistry) -> t.StrSequence:
        """Return promoted verbs in display order."""
        return tuple(sorted(registry.commands_by_verb))

    @staticmethod
    def make_registry_resolve_verb(
        registry: m.Tests.MakeRegistry, verb: str
    ) -> p.Result[str]:
        """Resolve one verb or alias to its canonical verb."""
        resolved = registry.aliases_by_name.get(verb, verb)
        if resolved not in registry.commands_by_verb:
            return r[str].fail(f"verb '{verb}' unknown")
        return r[str].ok(resolved)

    @staticmethod
    def make_registry_commands(
        registry: m.Tests.MakeRegistry, verb: str
    ) -> p.Result[t.MappingKV[str, m.Tests.MakeCommand]]:
        """Return commands registered for one verb."""
        resolved = FlextTestsMakeContractUtilitiesMixin.make_registry_resolve_verb(
            registry, verb
        )
        if resolved.failure:
            return r[t.MappingKV[str, m.Tests.MakeCommand]].fail(
                resolved.error or "verb unknown"
            )
        return r[t.MappingKV[str, m.Tests.MakeCommand]].ok(
            registry.commands_by_verb[resolved.value]
        )

    @staticmethod
    def make_registry_command(
        registry: m.Tests.MakeRegistry, verb: str, what: str
    ) -> p.Result[m.Tests.MakeCommand]:
        """Return one command by verb and WHAT value."""
        commands_result = FlextTestsMakeContractUtilitiesMixin.make_registry_commands(
            registry, verb
        )
        if commands_result.failure:
            return r[m.Tests.MakeCommand].fail(commands_result.error or "verb unknown")
        commands = commands_result.value
        command = commands.get(what)
        if command is None:
            valid = " ".join(sorted(commands))
            return r[m.Tests.MakeCommand].fail(
                f"WHAT='{what}' invalido para {verb}. Validos: {valid}"
            )
        return r[m.Tests.MakeCommand].ok(command)

    @staticmethod
    def make_registry_aliases_for(
        registry: m.Tests.MakeRegistry, verb: str
    ) -> t.StrSequence:
        """Return aliases that resolve to one canonical verb."""
        return tuple(
            sorted(
                alias
                for alias, target in registry.aliases_by_name.items()
                if target == verb
            )
        )


__all__: list[str] = ["FlextTestsMakeContractUtilitiesMixin"]
