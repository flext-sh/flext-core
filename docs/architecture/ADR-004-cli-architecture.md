# ADR-004: CLI Architecture

**Status**: Proposed
**Date**: 2025-06-28
**Based on**: Real code analysis of CLI implementations across flx-meltano-enterprise

## Context

The CLI functionality in flx-meltano-enterprise is scattered across multiple locations with different approaches. We need a unified CLI architecture that provides a consistent interface for all FLX platform operations.

### Current CLI Implementation Status

| Location                       | Purpose              | Implementation               |
| ------------------------------ | -------------------- | ---------------------------- |
| `src/flx_cli/cli.py`           | Main CLI entry       | Click-based, basic structure |
| `src/flx_core/application/cli` | Core CLI commands    | Command pattern              |
| Various `cli.py` files         | Module-specific CLIs | Inconsistent patterns        |
| `algar-oud-mig/cli.py`         | Project CLI          | Standalone implementation    |

## Decision

We will create a unified **flx-cli** module that:

1. **Consolidates all CLI functionality** into a single, well-organized module
2. **Uses Click framework** for consistent command structure
3. **Integrates with flx-core CommandBus** for business logic execution
4. **Provides plugin architecture** for extending commands

### Architecture

```
flx-cli/
├── src/
│   └── flx_cli/
│       ├── __init__.py
│       ├── cli.py              # Main entry point
│       ├── commands/
│       │   ├── __init__.py
│       │   ├── pipeline.py     # Pipeline management commands
│       │   ├── project.py      # Project management commands
│       │   ├── plugin.py       # Plugin management commands
│       │   ├── auth.py         # Authentication commands
│       │   ├── config.py       # Configuration commands
│       │   └── system.py       # System/admin commands
│       ├── plugins/
│       │   ├── __init__.py
│       │   └── loader.py       # CLI plugin loader
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── output.py       # Output formatting
│       │   ├── config.py       # CLI configuration
│       │   └── completion.py   # Shell completion
│       └── templates/
│           └── commands/       # Command templates
```

### Command Structure

```bash
# Hierarchical command structure
flx [global-options] <command> [command-options] <subcommand> [args]

# Examples:
flx --debug pipeline create my-pipeline --template github-to-postgres
flx project list --format json
flx auth login --username admin
flx plugin install tap-github --version 1.0.0
flx system health --verbose
```

### Integration with Core

```python
# CLI command implementation pattern
import click
from flx_core.application import CommandBus
from flx_core.domain.commands import CreatePipelineCommand

@click.command()
@click.argument('name')
@click.option('--template', help='Pipeline template to use')
@click.pass_context
def create(ctx, name: str, template: str):
    """Create a new pipeline."""
    command_bus: CommandBus = ctx.obj['command_bus']

    command = CreatePipelineCommand(
        name=name,
        template=template,
        user_id=ctx.obj['user_id']
    )

    result = command_bus.execute(command)

    if result.is_success:
        click.echo(f"✅ Pipeline '{name}' created successfully")
        click.echo(f"ID: {result.value.id}")
    else:
        click.echo(f"❌ Failed to create pipeline: {result.error}", err=True)
        ctx.exit(1)
```

## Consequences

### Positive

1. **Unified Interface**: Single entry point for all platform operations
2. **Consistent UX**: Standardized command structure and output
3. **Extensibility**: Plugin architecture for custom commands
4. **Testability**: Separation of CLI logic from business logic
5. **Documentation**: Auto-generated help from Click decorators
6. **Shell Integration**: Completion support for better UX

### Negative

1. **Migration Effort**: Need to consolidate scattered CLIs
2. **Breaking Changes**: Existing CLI users need to adapt
3. **Complexity**: Plugin system adds architectural overhead

## Implementation Plan

### Phase 1: Core Structure (Week 1)

- Create flx-cli module structure
- Implement base command groups
- Set up Click application with plugins

### Phase 2: Command Migration (Week 2)

- Migrate pipeline commands
- Migrate project commands
- Migrate authentication commands
- Migrate system commands

### Phase 3: Advanced Features (Week 3)

- Implement plugin loader
- Add shell completion
- Create output formatters
- Add interactive mode

### Phase 4: Integration (Week 4)

- Update all modules to use unified CLI
- Create migration guide
- Update documentation

## Technical Details

### Plugin Architecture

```python
# CLI plugin interface
from abc import ABC, abstractmethod
from click import Group

class CliPlugin(ABC):
    """Base class for CLI plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass

    @abstractmethod
    def get_commands(self) -> Group:
        """Return Click command group."""
        pass
```

### Output Formatting

```python
# Flexible output formatting
from enum import Enum
from typing import Any
import json
import yaml
from rich.console import Console
from rich.table import Table

class OutputFormat(Enum):
    TEXT = "text"
    JSON = "json"
    YAML = "yaml"
    TABLE = "table"

class OutputFormatter:
    """Format CLI output based on user preference."""

    def format(self, data: Any, format: OutputFormat) -> str:
        if format == OutputFormat.JSON:
            return json.dumps(data, indent=2)
        elif format == OutputFormat.YAML:
            return yaml.dump(data)
        elif format == OutputFormat.TABLE:
            return self._format_table(data)
        else:
            return self._format_text(data)
```

### Configuration Management

```python
# CLI configuration with multiple sources
from pathlib import Path
import toml

class CliConfig:
    """Manage CLI configuration."""

    def __init__(self):
        self.config_paths = [
            Path.home() / ".config" / "flx" / "cli.toml",
            Path.cwd() / ".flx" / "cli.toml",
            Path("cli.toml")
        ]

    def load(self) -> dict:
        """Load configuration from multiple sources."""
        config = {}
        for path in self.config_paths:
            if path.exists():
                config.update(toml.load(path))
        return config
```

## Success Metrics

- All platform operations accessible via CLI
- < 100ms startup time
- 95%+ command test coverage
- Plugin system supports 10+ extensions
- Shell completion for all commands
- Consistent error handling and output

## Security Considerations

1. **Authentication**: Secure token storage
2. **Authorization**: Command-level permissions
3. **Audit**: Log all CLI operations
4. **Input Validation**: Sanitize all inputs
5. **Secure Defaults**: Safe configuration defaults

## References

- [Click Documentation](https://click.palletsprojects.com/)
- [flx-core CommandBus](../../../flx-core/src/flx_core/application/commands.py)
- [Current CLI Implementations](../../../src/flx_cli/)
- [Python Fire Alternative](https://github.com/google/python-fire) (considered but rejected)
