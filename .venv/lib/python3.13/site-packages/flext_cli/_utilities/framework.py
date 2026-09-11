"""Private Click/Typer adapter behind the public FLEXT CLI facade."""

from __future__ import annotations

import sys
from collections.abc import Callable
from contextvars import ContextVar
from inspect import Parameter
from types import EllipsisType, GenericAlias
from typing import TYPE_CHECKING, Never

import click
import typer
from typer.models import OptionInfo
from typer.testing import CliRunner

# mro-j47u (codex): consume every public facade through the package root.
from flext_cli import c, e, r, t

if TYPE_CHECKING:
    from flext_cli import m, p

    # typer ships a vendored click, so the exceptions it raises are NOT instances
    # of the top-level `click` package's classes. `typer.BadParameter` is exported
# from that vendored module, so walking to its ClickException ancestor names
# the base class through typer's PUBLIC surface -- no private module access.
_TYPER_CLICK_EXCEPTION: type[Exception] = next(
    base for base in typer.BadParameter.__mro__ if base.__name__ == "ClickException"
)


class _TyperApplication:
    """Private application implementation hidden behind ``p.Cli.Application``."""

    __slots__ = ("_app", "_name")

    def __init__(self, app: typer.Typer, *, name: str | None) -> None:
        self._app = app
        self._name = name

    @property
    def name(self) -> str | None:
        """The configured application name."""
        return self._name

    @property
    def backend(self) -> typer.Typer:
        """The backend object inside this private adapter module."""
        return self._app

    def callback(
        self,
    ) -> Callable[[Callable[..., t.JsonPayload]], Callable[..., t.JsonPayload]]:
        """Return the private framework callback decorator."""
        return self._app.callback()

    def command[TCommand: Callable[..., t.JsonPayload]](
        self, name: str | None = None, *, help_text: str | None = None
    ) -> Callable[[TCommand], TCommand]:
        """Return a typed command decorator through the neutral contract."""
        return self._app.command(name, help=help_text)

    def add_typer(self, group: p.Cli.Application, *, name: str) -> None:
        """Attach another adapter-owned application as a child group."""
        if not isinstance(group, _TyperApplication):
            msg = "CLI group was not created by flext_cli"
            raise TypeError(msg)
        self._app.add_typer(group.backend, name=name)


class _ClickCommand:
    """Private command implementation satisfying ``p.Cli.ExternalCommand``."""

    __slots__ = ("_command",)

    def __init__(self, command: p.Cli.ExternalCommand) -> None:
        self._command = command

    def main(
        self,
        args: list[str] | None = None,
        prog_name: str | None = None,
        *,
        standalone_mode: bool = True,
    ) -> t.JsonPayload:
        """Execute and validate the backend command result at the boundary."""
        result = self._command.main(
            args=args, prog_name=prog_name, standalone_mode=standalone_mode
        )
        return t.Cli.JSON_VALUE_ADAPTER.validate_python(result)


class FlextCliUtilitiesFramework:
    """Single adapter owning all Click/Typer runtime interaction."""

    _active_execution: ContextVar[bool] = ContextVar(
        "flext_cli_active_execution", default=False
    )
    _active_failure: ContextVar[p.Result[t.Cli.ResultValue] | None] = ContextVar(
        "flext_cli_active_failure", default=None
    )

    @classmethod
    def framework_exit_result[TResult: t.Cli.ResultValue](
        cls, result: p.Result[TResult]
    ) -> bool:
        """Exit with a captured Result, or report a direct framework invocation."""
        # NOTE (multi-agent): this outer framework boundary is the single point
        # that exposes a failed Result to the user before the process exits;
        # every service layer keeps the canonical Result intact up to here.
        _ = r.require_error(result)
        if not cls._active_execution.get():
            return False
        from flext_cli import settings, u

        u.Cli.commands_emit_result_error(result, verbose=settings.cli_verbose)
        cls._active_failure.set(r[t.Cli.ResultValue].from_failure(result))
        return cls.framework_exit(code=c.Cli.EXIT_CODE_FAILURE)

    @staticmethod
    def _unwrap(application: p.Cli.Application) -> _TyperApplication:
        """Return the private application or fail on a foreign implementation."""
        if not isinstance(application, _TyperApplication):
            msg = "CLI application was not created by flext_cli"
            raise TypeError(msg)
        return application

    @staticmethod
    def _exit_code_result(exit_code: int) -> p.Result[bool]:
        """Normalize one framework exit code through the exception facade."""
        if exit_code == c.Cli.EXIT_CODE_SUCCESS:
            return r[bool].ok(True)
        return e.fail_operation(
            c.Cli.OP_EXECUTE_APPLICATION,
            c.Cli.ERR_EXIT_WITH_CODE.format(exit_code=exit_code),
            result_type=r[bool],
        )

    @classmethod
    def framework_create_app(
        cls, *, name: str | None, help_text: str, add_completion: bool = True
    ) -> p.Cli.Application:
        """Create one private Typer application behind the neutral protocol."""
        return _TyperApplication(
            typer.Typer(name=name, help=help_text, add_completion=add_completion),
            name=name,
        )

    @classmethod
    def framework_add_group(
        cls, application: p.Cli.Application, *, name: str, group: p.Cli.Application
    ) -> None:
        """Attach one private application group."""
        cls._unwrap(application).add_typer(group, name=name)

    @classmethod
    def framework_register_callback(
        cls, application: p.Cli.Application, callback: t.Cli.CliCommand
    ) -> None:
        """Register one application callback."""
        _ = cls._unwrap(application).callback()(callback)

    @classmethod
    def framework_register_command(
        cls,
        application: p.Cli.Application,
        *,
        name: str,
        help_text: str,
        command: t.Cli.CliCommand,
    ) -> None:
        """Register one named command."""
        _ = cls._unwrap(application).command(name, help_text=help_text)(command)

    @staticmethod
    def framework_build_parameter(
        field_name: str, annotation: type | GenericAlias, spec: m.Cli.OptionSpec
    ) -> Parameter:
        """Build one inspect parameter with a private Typer option default."""
        option_default: t.Cli.CliValue | EllipsisType | None = (
            ... if spec.required else spec.default
        )
        option = OptionInfo(
            default=option_default,
            param_decls=list(spec.declarations),
            help=spec.help_text or None,
        )
        return Parameter(
            field_name,
            kind=Parameter.KEYWORD_ONLY,
            default=option,
            annotation=annotation,
        )

    @classmethod
    def framework_execute(
        cls,
        application: p.Cli.Application,
        *,
        prog_name: str,
        args: t.StrSequence | None = None,
    ) -> p.Result[bool]:
        """Execute one application and normalize every framework exit path."""
        cli_args = list(args) if args is not None else sys.argv[1:]
        private_application = cls._unwrap(application)
        command = typer.main.get_command(private_application.backend)
        original_argv = sys.argv.copy()
        token = cls._active_execution.set(True)
        failure_token = cls._active_failure.set(None)
        captured_failure: p.Result[t.Cli.ResultValue] | None = None
        try:
            sys.argv = [prog_name, *cli_args]
            exit_result = command.main(
                args=cli_args, prog_name=prog_name, standalone_mode=False
            )
        # typer vendors its own click, so a usage error raised while resolving a
        # command is typer._click.exceptions.UsageError -- a DIFFERENT class from
        # the click.ClickException imported here. Catching only the latter let an
        # unknown command escape every handler and propagate out of the facade.
        except (click.ClickException, _TYPER_CLICK_EXCEPTION) as exc:
            return e.fail_validation(error=exc, result_type=r[bool])
        except typer.Abort as exc:
            return e.fail_operation(
                c.Cli.OP_EXECUTE_APPLICATION, exc, result_type=r[bool]
            )
        except typer.Exit as exc:
            if (failure := cls._active_failure.get()) is not None:
                return r[bool].from_failure(failure)
            return cls._exit_code_result(exc.exit_code)
        except SystemExit as exc:
            if (failure := cls._active_failure.get()) is not None:
                return r[bool].from_failure(failure)
            exit_code = (
                exc.code if isinstance(exc.code, int) else c.Cli.EXIT_CODE_FAILURE
            )
            return cls._exit_code_result(exit_code)
        except c.CATCHABLE_RUNTIME_EXCEPTIONS as exc:
            return e.fail_operation(
                c.Cli.OP_EXECUTE_APPLICATION, exc, result_type=r[bool]
            )
        finally:
            sys.argv = original_argv
            captured_failure = cls._active_failure.get()
            cls._active_failure.reset(failure_token)
            cls._active_execution.reset(token)
        if captured_failure is not None:
            return r[bool].from_failure(captured_failure)
        if (
            isinstance(exit_result, int)
            and not isinstance(exit_result, bool)
            and exit_result != c.Cli.EXIT_CODE_SUCCESS
        ):
            return cls._exit_code_result(exit_result)
        return r[bool].ok(True)

    @classmethod
    def framework_execute_external(
        cls,
        command: p.Cli.ExternalCommand,
        *,
        prog_name: str,
        args: t.StrSequence | None = None,
    ) -> p.Result[bool]:
        """Execute a foreign Click-compatible command inside the boundary."""
        try:
            # mro-wkii.17 (codex): normalize the public immutable sequence once
            # at the private Click boundary instead of weakening its protocol.
            exit_result = command.main(
                args=list(args) if args is not None else None,
                prog_name=prog_name,
                standalone_mode=False,
            )
        except click.ClickException as exc:
            return e.fail_validation(error=exc, result_type=r[bool])
        except click.Abort as exc:
            return e.fail_operation(
                c.Cli.OP_EXECUTE_APPLICATION, exc, result_type=r[bool]
            )
        except SystemExit as exc:
            exit_code = (
                exc.code if isinstance(exc.code, int) else c.Cli.EXIT_CODE_FAILURE
            )
            return cls._exit_code_result(exit_code)
        if (
            isinstance(exit_result, int)
            and not isinstance(exit_result, bool)
            and exit_result != c.Cli.EXIT_CODE_SUCCESS
        ):
            return cls._exit_code_result(exit_result)
        return r[bool].ok(True)

    @classmethod
    def framework_external_command(
        cls, application: p.Cli.Application
    ) -> p.Cli.ExternalCommand:
        """Expose an adapter-owned application through the command protocol."""
        return _ClickCommand(typer.main.get_command(cls._unwrap(application).backend))

    @classmethod
    def framework_invoke(
        cls,
        application: p.Cli.Application,
        *,
        args: t.StrSequence | None = None,
        charset: str = c.Cli.ENCODING_DEFAULT,
        env: t.StrMapping | None = None,
    ) -> m.Cli.InvocationResult:
        """Invoke one application through the real framework test runner.

        Captured output renders unstyled so text assertions stay
        presentation-free and stable in styled environments: the runner sets
        ``NO_COLOR`` and neutralizes the terminal-forcing signals typer
        consults instead of it (``GITHUB_ACTIONS``, ``FORCE_COLOR``,
        ``PY_COLORS``); an explicit caller ``env`` entry still wins.
        """
        from flext_cli import m

        runner_env: dict[str, str | None] = {
            "NO_COLOR": "1",
            "GITHUB_ACTIONS": "",
            "FORCE_COLOR": "",
            "PY_COLORS": "",
        }
        if env is not None:
            runner_env.update(env)
        runner = CliRunner(charset=charset, env=runner_env)
        private_application = cls._unwrap(application)
        result = runner.invoke(
            private_application.backend, args=list(args) if args is not None else None
        )
        return m.Cli.InvocationResult(
            exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr
        )

    @classmethod
    def framework_exit(cls, code: int = c.Cli.EXIT_CODE_SUCCESS) -> Never:
        """Exit through Typer only while an adapter-owned execution is active."""
        if cls._active_execution.get():
            raise typer.Exit(code=code)
        raise SystemExit(code)


__all__: list[str] = ["FlextCliUtilitiesFramework"]
