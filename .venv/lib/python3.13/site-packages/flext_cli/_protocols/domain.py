"""Higher-level CLI structural contracts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

from flext_cli import t
from flext_cli._protocols.base import FlextCliProtocolsBase


class FlextCliProtocolsDomain:
    """CLI domain protocols layered on top of base callable contracts."""

    type ResultRouteHandler = Callable[..., FlextCliProtocolsBase.ErasedCommandResult]

    @runtime_checkable
    class JsonValueProcessor(Protocol):
        """Protocol for JSON-compatible value processors."""

        def __call__(self, value: t.JsonValue) -> t.JsonValue:
            """Transform one JSON-compatible value."""
            ...

    class YamlAnchorNode(Protocol):
        """ruamel.yaml node surface that can carry YAML anchors.

        NOTE (multi-agent): mirrors the minimal ``yaml_set_anchor`` contract of
        ruamel ``CommentedBase``. Consumed through a ``TypeGuard`` + ``hasattr``
        check (not ``isinstance``) so leaf modules never import ruamel classes.
        """

        def yaml_set_anchor(self, value: str | None) -> None:
            """Set or clear the YAML anchor on the node."""
            ...

    @runtime_checkable
    class ModelCommandHandler[TParams: t.Cli.ModelLike](Protocol):
        """Protocol for model-driven CLI command execution."""

        def __call__(self, params: TParams, /) -> t.JsonValue:
            """Execute one model-backed CLI command and return its normalized value."""
            ...

    @runtime_checkable
    class CommandEntry(Protocol):
        """Protocol for command registry entries."""

        name: str
        # mro-j47u (codex): callable behavior remains in the p facade.
        handler: FlextCliProtocolsBase.CliCommandWrapper

    @runtime_checkable
    class ResultCommandRoute(Protocol):
        """Protocol for declarative result-route registration."""

        # mro-j47u (codex): read-only properties preserve frozen-model covariance.
        @property
        def name(self) -> str:
            """The command name."""
            ...

        @property
        def help_text(self) -> str:
            """The user-facing help text."""
            ...

        @property
        def model_cls(self) -> t.ModelClass[t.Cli.ModelLike]:
            """The validated input model class."""
            ...

        @property
        def handler(self) -> FlextCliProtocolsDomain.ResultRouteHandler:
            """The type-erased result handler."""
            ...

        @property
        def success_message(self) -> str | None:
            """The static success message, when configured."""
            ...

        @property
        def success_formatter(
            self,
        ) -> t.Cli.SuccessMessageFormatter[t.Cli.ResultValue] | None:
            """The dynamic success formatter, when configured."""
            ...

        @property
        def success_type(self) -> t.Cli.MessageType:
            """The success output style."""
            ...

    @runtime_checkable
    class DeclarativeRuleType[TRule](Protocol):
        """Class contract for one settings-backed declarative rule implementation."""

        RULE_MATCHERS: t.Cli.RuleMatchers

        def __call__(self, settings: t.JsonMapping, /) -> TRule:
            """Instantiate one runtime rule from one validated rule definition."""
            ...

    @runtime_checkable
    class DeclarativeFileRuleType[TRule](Protocol):
        """Class contract for one no-arg declarative file-rule implementation."""

        RULE_MATCHERS: t.Cli.RuleMatchers

        def __call__(self) -> TRule:
            """Instantiate one file rule without extra runtime settings."""
            ...

    @runtime_checkable
    class SummaryStats(Protocol):
        """Workspace orchestration summary payload contract."""

        @property
        def verb(self) -> str:
            """Verb label for the summary block."""
            ...

        @property
        def total(self) -> int:
            """Total processed items."""
            ...

        @property
        def success(self) -> int:
            """Successful items."""
            ...

        @property
        def failed(self) -> int:
            """Failed items."""
            ...

        @property
        def skipped(self) -> int:
            """Skipped items."""
            ...

        @property
        def elapsed(self) -> float:
            """Elapsed time in seconds."""
            ...

    @runtime_checkable
    class ProjectFailureInfo(Protocol):
        """Per-project failure descriptor for verbose diagnostics."""

        @property
        def project(self) -> str:
            """Project name."""
            ...

        @property
        def elapsed(self) -> float:
            """Elapsed time in seconds."""
            ...

        @property
        def error_count(self) -> int:
            """Total project errors."""
            ...

        @property
        def log_path(self) -> Path:
            """Path to the project log."""
            ...

        @property
        def max_show(self) -> int:
            """Maximum errors to render."""
            ...

        @property
        def errors(self) -> t.SequenceOf[str]:
            """Rendered error excerpt lines."""
            ...


__all__: list[str] = ["FlextCliProtocolsDomain"]
