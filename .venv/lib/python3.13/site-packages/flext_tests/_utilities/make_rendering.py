"""Make command rendering utilities for flext-tests."""

from __future__ import annotations

from collections.abc import Iterable

from flext_tests import c, m, p, r, t
from flext_tests._utilities.make_registry import FlextTestsMakeRegistryUtilitiesMixin


class FlextTestsMakeRenderingUtilitiesMixin(FlextTestsMakeRegistryUtilitiesMixin):
    """Render Make command help and output."""

    @staticmethod
    def make_format_params_inline(params: Iterable[m.Tests.MakeParam]) -> str:
        """Render command params in one compact inline form."""
        parts: list[str] = []
        for param in params:
            suffix = "*" if param.required else ""
            detail: list[str] = []
            if param.default:
                detail.append(f"default={param.default}")
            if param.choices:
                detail.append(f"choices={','.join(param.choices)}")
            rendered = f"{param.name}{suffix}"
            if detail:
                rendered = f"{rendered}({';'.join(detail)})"
            parts.append(rendered)
        return ", ".join(parts)

    @staticmethod
    def make_mutation_marker(command: m.Tests.MakeCommand) -> str:
        """Return the compact mutation marker for verb help."""
        if command.mutates:
            return " [mutates]"
        if command.mutates_when:
            return " [conditional-mutates]"
        return ""

    @staticmethod
    def make_mutation_label(command: m.Tests.MakeCommand) -> str:
        """Return the reader-facing mutation label for command help."""
        if command.mutates:
            return "sim"
        if command.mutates_when:
            return "condicional"
        return "nao"

    @staticmethod
    def make_format_mutation_conditions(
        conditions: Iterable[m.Tests.MakeMutationCondition],
    ) -> str:
        """Render conditional mutation predicates compactly."""
        return "; ".join(
            f"{condition.name}={('|'.join(condition.values))}"
            for condition in conditions
        )

    @staticmethod
    def make_example_for(command: m.Tests.MakeCommand, requested_verb: str) -> str:
        """Return the example adjusted for an alias-preserving verb."""
        example: str = command.example
        canonical = f"make {command.verb}"
        requested = f"make {requested_verb}"
        if example.startswith(canonical):
            return requested + example[len(canonical) :]
        return example

    @staticmethod
    def make_render_global_help(registry: m.Tests.MakeRegistry) -> str:
        """Render top-level dispatcher help."""
        lines = ["flext - make <verb> WHAT=<action> [PARAM=value ...]", ""]
        for verb in FlextTestsMakeRegistryUtilitiesMixin.make_registry_verbs(registry):
            command = registry.commands_by_verb[verb][c.Tests.MAKE_DEFAULT_COMMAND]
            aliases = FlextTestsMakeRegistryUtilitiesMixin.make_registry_aliases_for(
                registry, verb
            )
            suffix = f" (alias: {', '.join(aliases)})" if aliases else ""
            lines.append(f"  {verb:14} [{command.domain:12}] {command.summary}{suffix}")
        lines.extend((
            "",
            "Usage: make <verb> to list the available WHAT actions.",
            "Usage: make <verb> WHAT=<action> to execute.",
            "Usage: make <verb> WHAT=<verb>/<action> for action help.",
            "Mutating commands require APPLY=Y.",
            "",
            "Custom hooks: define pre-<verb>, post-<verb>, pre-<verb>-<what>, or",
            "post-<verb>-<what> in workspace_custom.mk to append work at the start",
            "or end of any verb, for all or some WHATs.",
        ))
        return "\n".join(lines)

    @staticmethod
    def make_render_verb_help(
        registry: m.Tests.MakeRegistry, requested_verb: str
    ) -> p.Result[str]:
        """Render help for one promoted verb."""
        verb_result = FlextTestsMakeRegistryUtilitiesMixin.make_registry_resolve_verb(
            registry, requested_verb
        )
        if verb_result.failure:
            return r[str].fail(verb_result.error or "verb unknown")
        verb = verb_result.value
        aliases = FlextTestsMakeRegistryUtilitiesMixin.make_registry_aliases_for(
            registry, verb
        )
        alias_suffix = f" (alias: {', '.join(aliases)})" if aliases else ""
        lines = [
            f"make {requested_verb} WHAT=<WHAT>{alias_suffix}",
            "",
            "WHAT available:",
        ]
        commands = registry.commands_by_verb[verb]
        for what, command in sorted(commands.items()):
            marker = FlextTestsMakeRenderingUtilitiesMixin.make_mutation_marker(command)
            lines.append(f"  {what:20} [{command.domain:12}] {command.summary}{marker}")
        command_params = [
            (what, command)
            for what, command in sorted(commands.items())
            if command.params
        ]
        if command_params:
            lines.extend(("", "Opcoes por WHAT:"))
            for what, command in command_params:
                rendered = (
                    FlextTestsMakeRenderingUtilitiesMixin.make_format_params_inline(
                        command.params
                    )
                )
                lines.append(f"  {what:20} {rendered}")
            lines.extend((
                "",
                "Detalhe de uma action:",
                f"  make {requested_verb} WHAT={requested_verb}/<WHAT>",
                f"  make {requested_verb} WHAT=<WHAT> OPTIONS=Y",
            ))
        rules = sorted({
            rule for command in commands.values() for rule in command.rules
        })
        if rules:
            lines.extend(("", "Regras:"))
            lines.extend(f"  - {rule}" for rule in rules)
        examples = sorted({
            FlextTestsMakeRenderingUtilitiesMixin.make_example_for(
                command, requested_verb
            )
            for command in commands.values()
        })
        if examples:
            lines.extend(("", "Exemplos:"))
            lines.extend(f"  {example}" for example in examples)
        return r[str].ok("\n".join(lines))

    @staticmethod
    def make_render_command_help(
        registry: m.Tests.MakeRegistry, requested_verb: str, what: str
    ) -> p.Result[str]:
        """Render help for one promoted command."""
        command_result = FlextTestsMakeRegistryUtilitiesMixin.make_registry_command(
            registry, requested_verb, what
        )
        if command_result.failure:
            return r[str].fail(command_result.error or "command unknown")
        command = command_result.value
        lines = [
            f"make {requested_verb} WHAT={what}",
            "",
            f"Dominio: {command.domain}",
            f"Mutaction: {FlextTestsMakeRenderingUtilitiesMixin.make_mutation_label(command)}",
        ]
        if command.mutates:
            lines.append("Sem APPLY=Y a execucao fica em dry-run.")
        elif command.mutates_when:
            conditions = (
                FlextTestsMakeRenderingUtilitiesMixin.make_format_mutation_conditions(
                    command.mutates_when
                )
            )
            lines.append(f"Mutaction condicional: {conditions}.")
        lines.extend(("", command.summary, command.description))
        if command.params:
            lines.extend(("", "Parameters:"))
            for param in command.params:
                required = " required" if param.required else ""
                default = f" default={param.default}" if param.default else ""
                choices = f" choices={','.join(param.choices)}" if param.choices else ""
                lines.append(
                    f"  {param.name:24} {param.help}{required}{default}{choices}"
                )
        if command.rules:
            lines.extend(("", "Regras:"))
            lines.extend(f"  - {rule}" for rule in command.rules)
        lines.extend((
            "",
            "Exemplo:",
            f"  {FlextTestsMakeRenderingUtilitiesMixin.make_example_for(command, requested_verb)}",
        ))
        return r[str].ok("\n".join(lines))

    @staticmethod
    def make_render_dry_run(
        command: m.Tests.MakeCommand,
        requested_verb: str,
        what: str,
        env: t.MappingKV[str, str],
    ) -> str:
        """Render dry-run output for one mutating command."""
        lines = [
            "DRY-RUN: nenhuma mutacao executada.",
            f"Comando: make {requested_verb} WHAT={what}",
            f"Domínio: {command.domain}",
            f"Resumo: {command.summary}",
            "Regra: command mutador exige APPLY=Y.",
        ]
        if command.params:
            lines.extend(("", "Current parameters:"))
            for param in command.params:
                value = FlextTestsMakeRegistryUtilitiesMixin.make_param_value(
                    param, command, env
                )
                shown = value or "<missing>"
                required = "required" if param.required else "opcional"
                choices = f" choices={','.join(param.choices)}" if param.choices else ""
                lines.append(
                    f"  {param.name:24} {shown:20} {required}{choices} - {param.help}"
                )
        lines.extend((
            "",
            "Execucao canonica:",
            f"  {FlextTestsMakeRenderingUtilitiesMixin.make_example_for(command, requested_verb)}",
        ))
        return "\n".join(lines)


__all__: list[str] = ["FlextTestsMakeRenderingUtilitiesMixin"]
