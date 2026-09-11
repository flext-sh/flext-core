"""Framework-neutral contracts for the CLI backend boundary."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

# mro-j47u (codex): consume the earlier local t facade through the package root.
from flext_cli import t


class FlextCliProtocolsFramework:
    """Structural contracts implemented by the private CLI framework adapter."""

    @runtime_checkable
    class Application(Protocol):
        """Opaque CLI application owned by the private framework adapter."""

        def callback(
            self,
        ) -> Callable[[Callable[..., t.JsonPayload]], Callable[..., t.JsonPayload]]:
            """Return the application callback decorator."""
            ...

        def command[TCommand: Callable[..., t.JsonPayload]](
            self, name: str | None = None, *, help_text: str | None = None
        ) -> Callable[[TCommand], TCommand]:
            """Return a named command decorator."""
            ...

        # mro-j47u (codex): match the sole adapter and mandatory named-group API.
        def add_typer(
            self, group: FlextCliProtocolsFramework.Application, *, name: str
        ) -> None:
            """Attach a child application under ``name``."""
            ...

    @runtime_checkable
    class ExternalCommand(Protocol):
        """Executable command contract used by framework integrations."""

        def main(
            self,
            # mro-wkii.17 (codex): Click's concrete boundary consumes a mutable
            # list; the public facade accepts any t.StrSequence and adapts once.
            args: list[str] | None = None,
            prog_name: str | None = None,
            *,
            standalone_mode: bool = True,
        ) -> t.JsonPayload:
            """Execute one command."""
            ...


__all__: list[str] = ["FlextCliProtocolsFramework"]
