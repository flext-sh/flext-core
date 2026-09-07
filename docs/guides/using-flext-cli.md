<!-- AUTO-GENERATED FILE — regenerate through `make gen APPLY=Y` from the workspace root. -->
<!-- Source of truth: `docs/guides/using-flext-cli.md`; adjust that source, never this projection. -->

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

```python

```

`flext_cli` reexports `d`, `e`, `h`, `r`, `x` from `flext_core`.

| Alias | Purpose |
| ------- | --------- |
| `c` | constants |
| `m` | models |
| `p` | protocols |
| `r` | result (reexported from `flext_core`) |
| `s` | service / runtime (`FlextCliServiceBase`) |
| `t` | typings |
| `u` | utilities |

**Important:** `s` is the service/runtime alias. CLI settings are accessed via `FlextCliSettings` (no short alias).

## Purpose

- Define CLI commands as Pydantic models.
- Let `FlextCliCli` convert model fields into Typer options.
- Keep output formatting, prompts, and runtime consistent across FLEXT CLI tools.

## Settings

Import the existing settings class; do not redefine it:

```python

```

If you need a project-specific subclass, extend `FlextSettings` (or `FlextCliSettings`) with `m.SettingsConfigDict`:

```python
from flext_core import FlextSettings, m


class FlextApiSettings(FlextSettings):
    model_config = m.SettingsConfigDict(env_prefix="FLEXT_API_", extra="ignore")
```

## Model-driven command

```python
from __future__ import annotations

from flext_cli import m, t
from flext_cli.services.cli import FlextCliCli
from flext_cli.settings import FlextCliSettings


settings = FlextCliSettings.fetch_global()


class GreetInput(m.BaseModel):
    name: str
    shout: bool = False


def greet_handler(model: GreetInput) -> t.JsonValue:
    message = f"Hello, {model.name}!"
    if model.shout:
        message = message.upper()
    return {"message": message}


command = FlextCliCli.model_command(
    model_cls=GreetInput, handler=greet_handler, settings=settings
)
```

**Common mistakes to avoid:**

- `FlextCliCli.build_model_command(...)` does not exist; use `FlextCliCli.model_command(...)`.
- `m.CliInput` / `m.CliOutput` do not exist; use plain `m.BaseModel` subclasses.

## Testing a command

```python
from typer.testing import CliRunner

runner = CliRunner()
result = runner.invoke(app, ["greet", "--name", "Ada"])
assert result.exit_code == 0
```

## Good practices

- Use plain `m.BaseModel` subclasses for command input.
- Read settings via `FlextCliSettings.fetch_global()`; `s` is the service/runtime alias.
- Avoid ad-hoc Typer functions and direct `u.Cli.print()`/`sys.exit()` in commands.

## Bad practices

```python
def main(name: str):  # ad-hoc command, no model
    u.Cli.print(f"Hello, {name}")
```

## Related

- `.agents/skills/using-flext-cli/SKILL.md`
- `.agents/skills/coding-standards/SKILL.md`
- `flext-cli/src/flext_cli/services/cli.py`
