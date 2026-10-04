<!-- AUTO-GENERATED FILE — regenerate through `make gen` from the workspace root. -->
<!-- Source of truth: `<workspace-root>/docs/guides/using-flext-cli.md`; adjust that workspace source, never this member projection. -->

# flext-core - Using flext-cli

> Project profile: `flext-core`

<!-- TOC START -->

- [Aliases](#aliases)
- [Purpose](#purpose)
- [Settings](#settings)
- [Model-driven command](#model-driven-command)
- [Testing a command](#testing-a-command)
- [Good practices](#good-practices)
- [Bad practices](#bad-practices)
- [Related](#related)

<!-- TOC END -->

`flext_cli` provides a unified Typer abstraction for model-driven CLI applications.

## Aliases

Import the aliases used by each example from the public `flext_cli` package root.

`flext_cli` reexports `d`, `e`, `h`, `r`, `x` from `flext_core`.

| Alias | Purpose                                   |
| ----- | ----------------------------------------- |
| `c`   | constants                                 |
| `m`   | models                                    |
| `p`   | protocols                                 |
| `r`   | result (reexported from `flext_core`)     |
| `s`   | service / runtime (`FlextCliServiceBase`) |
| `t`   | typings                                   |
| `u`   | utilities                                 |

**Important:** `s` is the service/runtime alias. CLI settings are accessed via
`FlextCliSettings` (no short alias).

## Purpose

- Define CLI commands as Pydantic models.
- Let `FlextCliCli` convert model fields into Typer options.
- Keep output formatting, prompts, and runtime consistent across FLEXT CLI tools.

## Settings

Import the existing settings class; do not redefine it:

```python
from flext_cli import FlextCliSettings

settings = FlextCliSettings.fetch_global()
if settings is not FlextCliSettings.fetch_global():
    message = "Expected cached settings singleton"
    raise RuntimeError(message)
```

If you need a project-specific subclass, extend `FlextSettings` (or `FlextCliSettings`)
with `m.SettingsConfigDict`:

```python
from flext_core import FlextSettings, m


class FlextApiSettings(FlextSettings):
    """Settings for the API demo with its own env prefix."""

    model_config = m.SettingsConfigDict(env_prefix="FLEXT_API_", extra="ignore")
```

## Model-driven command

```python
from __future__ import annotations

from flext_cli import FlextCliCli, FlextCliSettings, m, t

settings = FlextCliSettings.fetch_global()


class GreetInput(m.BaseModel):
    """Input model for the greeting command."""

    name: str
    shout: bool = False


def greet_handler(model: GreetInput) -> t.JsonValue:
    """Build the greeting payload for one input.

    Returns:
        The resulting ``t.JsonValue``.

    """
    message = f"Hello, {model.name}!"
    if model.shout:
        message = message.upper()
    return {"message": message}


command = FlextCliCli.model_command(
    model_cls=GreetInput, handler=greet_handler, settings=settings,
)
cli = FlextCliCli()
app = cli.create_app_with_common_params(name="greeting", help_text="Greeting commands")
cli.register_command(app, name="greet", help_text="Build a greeting", command=command)
```

**Common mistakes to avoid:**

- `FlextCliCli.build_model_command(...)` does not exist; use
  `FlextCliCli.model_command(...)`.
- `m.CliInput` / `m.CliOutput` do not exist; use plain `m.BaseModel` subclasses.

## Testing a command

Use `FlextCliCli.invoke_app` with the adapter-owned application, not Typer's `CliRunner`
directly. This independent example constructs and invokes a real model-backed command;
handlers return their value but do not automatically print it.

```python
from flext_cli import FlextCliCli, m


class GreetInput(m.BaseModel):
    """Input model for the greeting command."""

    name: str


def greet_handler(model: GreetInput) -> str:
    """Build the greeting line for one input.

    Returns:
        The resulting ``str``.

    """
    return f"Hello, {model.name}!"


def test_greet_command() -> None:
    """Invoke the greeting command end to end.

    Raises:
        RuntimeError: When the invocation or greeting misbehaves.

    """
    cli = FlextCliCli()
    app = cli.create_app_with_common_params(
        name="greeting", help_text="Greeting commands",
    )
    command = cli.model_command(model_cls=GreetInput, handler=greet_handler)
    cli.register_command(
        app, name="greet", help_text="Build a greeting", command=command,
    )
    invocation = cli.invoke_app(app, args=["greet", "--name", "Ada"])
    expected_greeting = "Hello, Ada!"
    if not invocation.success:
        message = "Expected CLI invocation success"
        raise RuntimeError(message)
    if invocation.value.exit_code != 0:
        message = "Expected zero CLI exit code"
        raise RuntimeError(message)
    if greet_handler(GreetInput(name="Ada")) != expected_greeting:
        message = "Unexpected handler greeting"
        raise RuntimeError(message)
```

## Good practices

- Use plain `m.BaseModel` subclasses for command input.
- Read settings via `FlextCliSettings.fetch_global()`; `s` is the service/runtime alias.
- Avoid ad-hoc Typer functions and direct `u.Cli.print()`/`sys.exit()` in commands.

## Bad practices

Do not replace the model command with an untyped ad-hoc handler or bypass the adapter
with direct printing and process termination. Register the model-backed command through
the public facade as shown above. A corrected handler consumes its declared input model
and returns its value:

```python
from flext_cli import m


class GreetInput(m.BaseModel):
    """Input model for the greeting command."""

    name: str


def greet_handler(model: GreetInput) -> str:
    """Build the greeting line for one input.

    Returns:
        The resulting ``str``.

    """
    return f"Hello, {model.name}!"


expected_greeting = "Hello, Ada!"
if greet_handler(GreetInput(name="Ada")) != expected_greeting:
    message = "Unexpected handler greeting"
    raise RuntimeError(message)
```

## Related

- `flext-cli/src/flext_cli/services/cli.py`
- Generated flext-core API reference
