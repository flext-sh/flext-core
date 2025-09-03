# CLI Commands CQRS Examples

**Version**: 1.0  
**Target**: CLI Developers  
**Framework**: Click, Typer  
**Complexity**: Intermediate

## 📋 Overview

This document provides practical examples of implementing FlextCommands CQRS patterns in command-line interfaces. It covers structured command processing, parameter validation, error handling, and integration with popular Python CLI frameworks.

## 🎯 Key Benefits

- ✅ **Structured Commands**: Commands as validated objects, not strings
- ✅ **Consistent Validation**: Business rules applied uniformly
- ✅ **Better Error Handling**: Structured error messages and codes
- ✅ **Command History**: Automatic logging and audit trails
- ✅ **Testability**: Easy unit testing of command logic

---

## 🚀 Click Integration Examples

### Pipeline Management CLI

#### Command Example: Execute Pipeline

```python
import click
from rich.console import Console
from rich.table import Table
from rich.progress import track
from flext_core import FlextCommands, FlextResult

console = Console()

# CQRS Command
class ExecutePipelineCommand(FlextCommands.Models.Command):
    pipeline_name: str
    environment: str = "development"
    dry_run: bool = False
    parameters: dict[str, object] = Field(default_factory=dict)
    max_retries: int = 3
    timeout_minutes: int = 30

    def validate_command(self) -> FlextResult[None]:
        return (
            self.require_field("pipeline_name", self.pipeline_name)
            .flat_map(lambda _: self._validate_environment())
            .flat_map(lambda _: self._validate_parameters())
        )

    def _validate_environment(self) -> FlextResult[None]:
        valid_envs = {"development", "staging", "production", "test"}
        if self.environment not in valid_envs:
            return FlextResult[None].fail(
                f"Invalid environment '{self.environment}'. Valid: {valid_envs}"
            )
        return FlextResult[None].ok(None)

    def _validate_parameters(self) -> FlextResult[None]:
        # Validate parameter types and values
        for key, value in self.parameters.items():
            if not isinstance(key, str) or len(key) == 0:
                return FlextResult[None].fail(f"Invalid parameter key: {key}")

        return FlextResult[None].ok(None)

# Command Handler
class ExecutePipelineHandler(FlextCommands.Handlers.CommandHandler[ExecutePipelineCommand, dict]):
    def __init__(self, pipeline_service: PipelineService, notification_service: NotificationService):
        super().__init__(handler_name="ExecutePipelineHandler")
        self.pipeline_service = pipeline_service
        self.notification_service = notification_service

    def handle(self, command: ExecutePipelineCommand) -> FlextResult[dict]:
        try:
            self.log_info("Starting pipeline execution",
                         pipeline=command.pipeline_name,
                         environment=command.environment,
                         dry_run=command.dry_run)

            # Validate pipeline exists
            pipeline = self.pipeline_service.get_pipeline(command.pipeline_name)
            if not pipeline:
                return FlextResult[dict].fail(
                    f"Pipeline '{command.pipeline_name}' not found",
                    error_code="PIPELINE_NOT_FOUND"
                )

            # Check pipeline status
            if not pipeline.is_ready_for_execution():
                return FlextResult[dict].fail(
                    f"Pipeline '{command.pipeline_name}' is not ready for execution",
                    error_code="PIPELINE_NOT_READY"
                )

            if command.dry_run:
                # Dry run - just validate and return plan
                execution_plan = self.pipeline_service.generate_execution_plan(
                    pipeline, command.parameters
                )
                return FlextResult[dict].ok({
                    "pipeline_name": command.pipeline_name,
                    "environment": command.environment,
                    "dry_run": True,
                    "execution_plan": execution_plan,
                    "estimated_duration_minutes": execution_plan.get("estimated_duration", 0)
                })

            # Execute pipeline
            execution = self.pipeline_service.execute_pipeline(
                pipeline=pipeline,
                environment=command.environment,
                parameters=command.parameters,
                max_retries=command.max_retries,
                timeout_minutes=command.timeout_minutes
            )

            # Send notification
            if execution.status == "started":
                self.notification_service.send_execution_started_notification(
                    pipeline_name=command.pipeline_name,
                    execution_id=execution.id,
                    user_id=command.user_id
                )

            self.log_info("Pipeline execution initiated",
                         pipeline=command.pipeline_name,
                         execution_id=execution.id,
                         status=execution.status)

            return FlextResult[dict].ok({
                "pipeline_name": command.pipeline_name,
                "execution_id": execution.id,
                "status": execution.status,
                "started_at": execution.started_at.isoformat(),
                "estimated_completion": execution.estimated_completion.isoformat() if execution.estimated_completion else None,
                "environment": command.environment,
                "dry_run": False
            })

        except Exception as e:
            self.log_error("Pipeline execution failed",
                          pipeline=command.pipeline_name,
                          error=str(e))
            return FlextResult[dict].fail(f"Pipeline execution failed: {e}")

# Click Command
@click.command()
@click.argument('pipeline_name')
@click.option('--env', '--environment', default='development',
              help='Execution environment (development, staging, production, test)')
@click.option('--dry-run', is_flag=True,
              help='Show execution plan without running')
@click.option('--param', '-p', multiple=True,
              help='Pipeline parameters in key=value format')
@click.option('--max-retries', type=int, default=3,
              help='Maximum number of retries on failure')
@click.option('--timeout', type=int, default=30,
              help='Timeout in minutes')
@click.option('--verbose', '-v', is_flag=True,
              help='Verbose output')
def execute_pipeline(pipeline_name: str, env: str, dry_run: bool,
                    param: tuple[str], max_retries: int, timeout: int, verbose: bool):
    """Execute a data pipeline with specified parameters.

    PIPELINE_NAME: Name of the pipeline to execute

    Examples:
        flext pipeline execute user-data-sync --env production
        flext pipeline execute etl-orders --dry-run --param source=api --param batch_size=1000
        flext pipeline execute data-validation --timeout 60 --max-retries 5
    """
    console = Console()

    try:
        # Parse parameters
        parameters = {}
        for p in param:
            if '=' not in p:
                console.print(f"[red]Invalid parameter format: {p}. Use key=value format.[/red]")
                raise click.Abort()

            key, value = p.split('=', 1)
            # Try to convert to appropriate type
            try:
                # Try int first
                if value.isdigit():
                    parameters[key] = int(value)
                # Try float
                elif '.' in value and value.replace('.', '').isdigit():
                    parameters[key] = float(value)
                # Try boolean
                elif value.lower() in ('true', 'false'):
                    parameters[key] = value.lower() == 'true'
                else:
                    parameters[key] = value
            except ValueError:
                parameters[key] = value

        # Create command
        command = ExecutePipelineCommand(
            pipeline_name=pipeline_name,
            environment=env,
            dry_run=dry_run,
            parameters=parameters,
            max_retries=max_retries,
            timeout_minutes=timeout
        )

        # Execute command
        if verbose:
            console.print(f"[blue]Executing pipeline: {pipeline_name}[/blue]")
            console.print(f"[blue]Environment: {env}[/blue]")
            console.print(f"[blue]Parameters: {parameters}[/blue]")

        result = command_bus.execute(command)

        # Handle result
        if result.success:
            data = result.value

            if dry_run:
                console.print("[green]✓ Dry run completed successfully[/green]")
                console.print(f"[yellow]Estimated duration: {data['estimated_duration_minutes']} minutes[/yellow]")

                # Display execution plan
                if verbose and 'execution_plan' in data:
                    table = Table(title="Execution Plan")
                    table.add_column("Step", style="cyan")
                    table.add_column("Action", style="magenta")
                    table.add_column("Duration", style="green")

                    for step in data['execution_plan'].get('steps', []):
                        table.add_row(
                            step.get('name', ''),
                            step.get('action', ''),
                            f"{step.get('duration_minutes', 0)}min"
                        )
                    console.print(table)
            else:
                console.print("[green]✓ Pipeline execution started successfully[/green]")
                console.print(f"[blue]Execution ID: {data['execution_id']}[/blue]")
                console.print(f"[blue]Status: {data['status']}[/blue]")
                console.print(f"[blue]Started at: {data['started_at']}[/blue]")

                if data.get('estimated_completion'):
                    console.print(f"[yellow]Estimated completion: {data['estimated_completion']}[/yellow]")

                console.print(f"\n[dim]Use 'flext pipeline status {data['execution_id']}' to check progress[/dim]")
        else:
            error_code = getattr(result, 'error_code', 'UNKNOWN_ERROR')

            # Handle specific error codes
            if error_code == 'PIPELINE_NOT_FOUND':
                console.print(f"[red]✗ Pipeline '{pipeline_name}' not found[/red]")
                console.print("[dim]Use 'flext pipeline list' to see available pipelines[/dim]")
            elif error_code == 'PIPELINE_NOT_READY':
                console.print(f"[red]✗ Pipeline '{pipeline_name}' is not ready for execution[/red]")
                console.print("[dim]Check pipeline configuration and dependencies[/dim]")
            elif error_code == 'VALIDATION_ERROR':
                console.print(f"[red]✗ Validation error: {result.error}[/red]")
            else:
                console.print(f"[red]✗ Error: {result.error}[/red]")

            raise click.Abort()

    except Exception as e:
        console.print(f"[red]✗ Command failed: {e}[/red]")
        if verbose:
            console.print_exception()
        raise click.Abort()
```

#### Query Example: Pipeline Status

```python
# Query for getting pipeline execution status
class GetPipelineStatusQuery(FlextCommands.Models.Query):
    execution_id: str | None = None
    pipeline_name: str | None = None
    include_logs: bool = False
    include_metrics: bool = False

    def validate_query(self) -> FlextResult[None]:
        if not self.execution_id and not self.pipeline_name:
            return FlextResult[None].fail("Either execution_id or pipeline_name must be provided")
        return FlextResult[None].ok(None)

class GetPipelineStatusHandler(FlextCommands.Handlers.QueryHandler[GetPipelineStatusQuery, dict]):
    def __init__(self, pipeline_service: PipelineService):
        super().__init__(handler_name="GetPipelineStatusHandler")
        self.pipeline_service = pipeline_service

    def handle(self, query: GetPipelineStatusQuery) -> FlextResult[dict]:
        try:
            if query.execution_id:
                # Get status by execution ID
                execution = self.pipeline_service.get_execution(query.execution_id)
                if not execution:
                    return FlextResult[dict].fail(
                        f"Execution {query.execution_id} not found",
                        error_code="EXECUTION_NOT_FOUND"
                    )
            else:
                # Get latest execution by pipeline name
                execution = self.pipeline_service.get_latest_execution(query.pipeline_name)
                if not execution:
                    return FlextResult[dict].fail(
                        f"No executions found for pipeline {query.pipeline_name}",
                        error_code="NO_EXECUTIONS_FOUND"
                    )

            # Build response
            status_data = {
                "execution_id": execution.id,
                "pipeline_name": execution.pipeline_name,
                "status": execution.status,
                "progress_percentage": execution.progress_percentage,
                "started_at": execution.started_at.isoformat(),
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "duration_seconds": execution.duration_seconds,
                "environment": execution.environment,
                "parameters": execution.parameters,
                "error_message": execution.error_message if execution.status == "failed" else None
            }

            # Include logs if requested
            if query.include_logs:
                logs = self.pipeline_service.get_execution_logs(execution.id)
                status_data["logs"] = [
                    {
                        "timestamp": log.timestamp.isoformat(),
                        "level": log.level,
                        "message": log.message,
                        "step": log.step
                    }
                    for log in logs
                ]

            # Include metrics if requested
            if query.include_metrics:
                metrics = self.pipeline_service.get_execution_metrics(execution.id)
                status_data["metrics"] = {
                    "records_processed": metrics.records_processed,
                    "records_failed": metrics.records_failed,
                    "data_size_mb": metrics.data_size_mb,
                    "cpu_usage_avg": metrics.cpu_usage_avg,
                    "memory_usage_avg": metrics.memory_usage_avg
                }

            return FlextResult[dict].ok(status_data)

        except Exception as e:
            return FlextResult[dict].fail(f"Failed to get pipeline status: {e}")

@click.command()
@click.argument('identifier')  # Can be execution_id or pipeline_name
@click.option('--logs', is_flag=True, help='Include execution logs')
@click.option('--metrics', is_flag=True, help='Include performance metrics')
@click.option('--follow', '-f', is_flag=True, help='Follow execution progress')
@click.option('--refresh', type=int, default=5, help='Refresh interval in seconds (with --follow)')
def status(identifier: str, logs: bool, metrics: bool, follow: bool, refresh: int):
    """Get pipeline execution status.

    IDENTIFIER: Execution ID or pipeline name

    Examples:
        flext pipeline status exec-123456789
        flext pipeline status user-data-sync --logs --metrics
        flext pipeline status exec-123456789 --follow --refresh 2
    """
    import time

    console = Console()

    try:
        # Determine if identifier is execution ID or pipeline name
        is_execution_id = len(identifier) > 10 and identifier.startswith(('exec-', 'run-'))

        query = GetPipelineStatusQuery(
            execution_id=identifier if is_execution_id else None,
            pipeline_name=identifier if not is_execution_id else None,
            include_logs=logs,
            include_metrics=metrics
        )

        def display_status():
            result = status_handler.handle(query)

            if result.success:
                data = result.value

                # Clear screen if following
                if follow:
                    console.clear()

                # Status header
                status_color = {
                    "running": "blue",
                    "completed": "green",
                    "failed": "red",
                    "pending": "yellow",
                    "cancelled": "red"
                }.get(data["status"], "white")

                console.print(f"[bold]Pipeline Status[/bold]")
                console.print(f"Pipeline: [cyan]{data['pipeline_name']}[/cyan]")
                console.print(f"Execution ID: [dim]{data['execution_id']}[/dim]")
                console.print(f"Status: [{status_color}]{data['status'].upper()}[/{status_color}]")
                console.print(f"Progress: {data['progress_percentage']}%")

                # Progress bar
                if data["status"] == "running" and data["progress_percentage"] > 0:
                    from rich.progress import Progress
                    with Progress() as progress:
                        task = progress.add_task("", total=100)
                        progress.update(task, completed=data["progress_percentage"])

                # Timing information
                console.print(f"Started: {data['started_at']}")
                if data["completed_at"]:
                    console.print(f"Completed: {data['completed_at']}")
                if data["duration_seconds"]:
                    console.print(f"Duration: {data['duration_seconds']} seconds")

                # Environment and parameters
                console.print(f"Environment: [yellow]{data['environment']}[/yellow]")
                if data["parameters"]:
                    console.print("Parameters:")
                    for key, value in data["parameters"].items():
                        console.print(f"  {key}: {value}")

                # Error message if failed
                if data["error_message"]:
                    console.print(f"[red]Error: {data['error_message']}[/red]")

                # Metrics
                if metrics and "metrics" in data:
                    m = data["metrics"]
                    console.print("\n[bold]Metrics:[/bold]")
                    console.print(f"Records processed: {m['records_processed']}")
                    console.print(f"Records failed: {m['records_failed']}")
                    console.print(f"Data size: {m['data_size_mb']} MB")
                    console.print(f"CPU usage (avg): {m['cpu_usage_avg']}%")
                    console.print(f"Memory usage (avg): {m['memory_usage_avg']} MB")

                # Logs
                if logs and "logs" in data:
                    console.print("\n[bold]Recent Logs:[/bold]")
                    for log in data["logs"][-10:]:  # Show last 10 logs
                        level_color = {
                            "INFO": "blue",
                            "WARNING": "yellow",
                            "ERROR": "red",
                            "DEBUG": "dim"
                        }.get(log["level"], "white")

                        console.print(f"[{level_color}]{log['timestamp']} [{log['level']}] {log['message']}[/{level_color}]")

                return data["status"]
            else:
                error_code = getattr(result, 'error_code', 'UNKNOWN_ERROR')
                if error_code == 'EXECUTION_NOT_FOUND':
                    console.print(f"[red]✗ Execution '{identifier}' not found[/red]")
                elif error_code == 'NO_EXECUTIONS_FOUND':
                    console.print(f"[red]✗ No executions found for pipeline '{identifier}'[/red]")
                else:
                    console.print(f"[red]✗ Error: {result.error}[/red]")
                return None

        # Display status
        current_status = display_status()

        # Follow mode
        if follow and current_status in ["running", "pending"]:
            console.print(f"\n[dim]Following execution... (press Ctrl+C to stop)[/dim]")
            try:
                while current_status in ["running", "pending"]:
                    time.sleep(refresh)
                    current_status = display_status()

                console.print(f"\n[green]✓ Execution finished with status: {current_status}[/green]")
            except KeyboardInterrupt:
                console.print(f"\n[yellow]Following stopped by user[/yellow]")

    except Exception as e:
        console.print(f"[red]✗ Command failed: {e}[/red]")
        raise click.Abort()
```

---

## 🔧 Advanced CLI Patterns

### Interactive Command with Validation

```python
class InitProjectCommand(FlextCommands.Models.Command):
    project_name: str
    project_type: str
    template: str = "basic"
    git_init: bool = True
    install_deps: bool = True
    target_directory: str = "."

    def validate_command(self) -> FlextResult[None]:
        # Validate project name
        if not self.project_name.isidentifier():
            return FlextResult[None].fail(
                "Project name must be a valid Python identifier"
            )

        # Validate project type
        valid_types = {"meltano", "dbt", "api", "cli", "web"}
        if self.project_type not in valid_types:
            return FlextResult[None].fail(
                f"Invalid project type. Valid options: {valid_types}"
            )

        # Validate template
        if self.project_type == "meltano":
            valid_templates = {"basic", "advanced", "enterprise"}
        elif self.project_type == "api":
            valid_templates = {"fastapi", "flask", "django"}
        else:
            valid_templates = {"basic", "advanced"}

        if self.template not in valid_templates:
            return FlextResult[None].fail(
                f"Invalid template '{self.template}' for project type '{self.project_type}'. "
                f"Valid templates: {valid_templates}"
            )

        return FlextResult[None].ok(None)

@click.command()
@click.option('--name', prompt='Project name', help='Name of the project')
@click.option('--type', 'project_type',
              type=click.Choice(['meltano', 'dbt', 'api', 'cli', 'web']),
              prompt='Project type', help='Type of project to create')
@click.option('--template', help='Project template to use')
@click.option('--git/--no-git', default=True, help='Initialize git repository')
@click.option('--deps/--no-deps', default=True, help='Install dependencies')
@click.option('--dir', 'target_directory', default='.', help='Target directory')
@click.option('--interactive', is_flag=True, help='Interactive mode with prompts')
def init(name: str, project_type: str, template: str, git: bool,
         deps: bool, target_directory: str, interactive: bool):
    """Initialize a new FLEXT project.

    Creates a new project with the specified type and template,
    optionally initializing git and installing dependencies.

    Examples:
        flext init --name my-pipeline --type meltano --template advanced
        flext init --interactive
    """
    console = Console()

    try:
        # Interactive mode
        if interactive:
            console.print("[bold blue]FLEXT Project Initialization[/bold blue]")

            # Get project details interactively
            name = click.prompt('Project name', type=str)
            project_type = click.prompt('Project type',
                                      type=click.Choice(['meltano', 'dbt', 'api', 'cli', 'web']))

            # Template selection based on project type
            if project_type == "meltano":
                template_choices = ["basic", "advanced", "enterprise"]
            elif project_type == "api":
                template_choices = ["fastapi", "flask", "django"]
            else:
                template_choices = ["basic", "advanced"]

            template = click.prompt('Template',
                                  type=click.Choice(template_choices),
                                  default=template_choices[0])

            git = click.confirm('Initialize git repository?', default=True)
            deps = click.confirm('Install dependencies?', default=True)
            target_directory = click.prompt('Target directory', default='.')

        # Set default template if not provided
        if not template:
            if project_type == "api":
                template = "fastapi"
            elif project_type == "meltano":
                template = "basic"
            else:
                template = "basic"

        # Create command
        command = InitProjectCommand(
            project_name=name,
            project_type=project_type,
            template=template,
            git_init=git,
            install_deps=deps,
            target_directory=target_directory
        )

        # Show configuration summary
        console.print("\n[bold]Project Configuration:[/bold]")
        table = Table(show_header=False, box=None)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Name", command.project_name)
        table.add_row("Type", command.project_type)
        table.add_row("Template", command.template)
        table.add_row("Directory", command.target_directory)
        table.add_row("Git Init", "Yes" if command.git_init else "No")
        table.add_row("Install Deps", "Yes" if command.install_deps else "No")

        console.print(table)

        if not click.confirm('\nProceed with project creation?', default=True):
            console.print("[yellow]Project creation cancelled[/yellow]")
            return

        # Execute command with progress tracking
        with console.status("[bold green]Creating project...") as status:
            result = command_bus.execute(command)

            if result.success:
                data = result.value
                console.print(f"\n[green]✓ Project '{name}' created successfully![/green]")
                console.print(f"[blue]Location: {data['project_path']}[/blue]")

                # Show next steps
                console.print("\n[bold]Next Steps:[/bold]")
                console.print(f"1. cd {data['project_path']}")

                if project_type == "meltano":
                    console.print("2. meltano install")
                    console.print("3. meltano run tap target")
                elif project_type == "api":
                    console.print("2. pip install -r requirements.txt")
                    console.print("3. python -m uvicorn main:app --reload")
                else:
                    console.print("2. Follow the README.md for setup instructions")

                if 'git_repository' in data:
                    console.print(f"4. Git repository initialized at: {data['git_repository']}")
            else:
                console.print(f"[red]✗ Project creation failed: {result.error}[/red]")
                raise click.Abort()

    except Exception as e:
        console.print(f"[red]✗ Command failed: {e}[/red]")
        raise click.Abort()
```

### Batch Processing Command

```python
class BatchProcessCommand(FlextCommands.Models.Command):
    operation: str  # "validate", "process", "deploy"
    file_patterns: list[str] = Field(default_factory=list)
    target_directory: str = "."
    parallel_workers: int = 4
    fail_fast: bool = False
    output_format: str = "table"  # "table", "json", "csv"

    def validate_command(self) -> FlextResult[None]:
        valid_operations = {"validate", "process", "deploy", "test"}
        if self.operation not in valid_operations:
            return FlextResult[None].fail(f"Invalid operation: {self.operation}")

        if self.parallel_workers < 1 or self.parallel_workers > 10:
            return FlextResult[None].fail("parallel_workers must be between 1 and 10")

        valid_formats = {"table", "json", "csv"}
        if self.output_format not in valid_formats:
            return FlextResult[None].fail(f"Invalid output format: {self.output_format}")

        return FlextResult[None].ok(None)

@click.command()
@click.argument('operation', type=click.Choice(['validate', 'process', 'deploy', 'test']))
@click.option('--pattern', '-p', multiple=True, help='File patterns to process')
@click.option('--dir', default='.', help='Target directory')
@click.option('--workers', type=int, default=4, help='Number of parallel workers')
@click.option('--fail-fast', is_flag=True, help='Stop on first error')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json', 'csv']),
              default='table', help='Output format')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def batch(operation: str, pattern: tuple[str], dir: str, workers: int,
          fail_fast: bool, output_format: str, verbose: bool):
    """Process multiple files in batch.

    OPERATION: Operation to perform on files

    Examples:
        flext batch validate --pattern "*.yml" --pattern "*.yaml"
        flext batch process --dir configs/ --workers 8 --fail-fast
        flext batch test --format json --verbose
    """
    console = Console()

    try:
        # Default patterns if none provided
        if not pattern:
            pattern_map = {
                "validate": ["*.yml", "*.yaml", "*.json"],
                "process": ["*.py", "*.sql"],
                "deploy": ["*.yml", "*.yaml"],
                "test": ["test_*.py", "*_test.py"]
            }
            pattern = pattern_map.get(operation, ["*"])

        command = BatchProcessCommand(
            operation=operation,
            file_patterns=list(pattern),
            target_directory=dir,
            parallel_workers=workers,
            fail_fast=fail_fast,
            output_format=output_format
        )

        if verbose:
            console.print(f"[blue]Operation: {operation}[/blue]")
            console.print(f"[blue]Patterns: {', '.join(pattern)}[/blue]")
            console.print(f"[blue]Directory: {dir}[/blue]")
            console.print(f"[blue]Workers: {workers}[/blue]")

        # Execute with progress tracking
        result = command_bus.execute(command)

        if result.success:
            data = result.value

            # Display results based on format
            if output_format == "table":
                table = Table(title=f"Batch {operation.title()} Results")
                table.add_column("File", style="cyan")
                table.add_column("Status", style="green")
                table.add_column("Duration", style="yellow")
                table.add_column("Message", style="white")

                for item in data["results"]:
                    status_color = "green" if item["status"] == "success" else "red"
                    table.add_row(
                        item["file_path"],
                        f"[{status_color}]{item['status']}[/{status_color}]",
                        f"{item['duration_ms']}ms",
                        item.get("message", "")
                    )
                console.print(table)

                # Summary
                console.print(f"\n[bold]Summary:[/bold]")
                console.print(f"Total files: {data['total_files']}")
                console.print(f"Successful: [green]{data['successful_files']}[/green]")
                console.print(f"Failed: [red]{data['failed_files']}[/red]")
                console.print(f"Duration: {data['total_duration_ms']}ms")

            elif output_format == "json":
                import json
                console.print(json.dumps(data, indent=2))

            elif output_format == "csv":
                import csv
                import io

                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(["file_path", "status", "duration_ms", "message"])

                for item in data["results"]:
                    writer.writerow([
                        item["file_path"],
                        item["status"],
                        item["duration_ms"],
                        item.get("message", "")
                    ])

                console.print(output.getvalue())
        else:
            console.print(f"[red]✗ Batch operation failed: {result.error}[/red]")
            raise click.Abort()

    except Exception as e:
        console.print(f"[red]✗ Command failed: {e}[/red]")
        if verbose:
            console.print_exception()
        raise click.Abort()
```

---

## 🧪 Testing CLI Commands

### Unit Testing Commands

```python
import pytest
from click.testing import CliRunner
from flext_core import FlextResult

class TestExecutePipelineCommand:
    """Test the execute pipeline command and handler."""

    def test_command_validation_success(self):
        """Test successful command validation."""
        command = ExecutePipelineCommand(
            pipeline_name="test-pipeline",
            environment="development"
        )

        result = command.validate_command()
        assert result.success

    def test_command_validation_invalid_environment(self):
        """Test command validation with invalid environment."""
        command = ExecutePipelineCommand(
            pipeline_name="test-pipeline",
            environment="invalid"
        )

        result = command.validate_command()
        assert result.is_failure
        assert "Invalid environment" in result.error

    @pytest.fixture
    def mock_pipeline_service(self, mocker):
        """Mock pipeline service for testing."""
        service = mocker.Mock()
        service.get_pipeline.return_value = mocker.Mock(
            is_ready_for_execution=lambda: True
        )
        service.execute_pipeline.return_value = mocker.Mock(
            id="exec-123",
            status="started",
            started_at=datetime.now(),
            estimated_completion=None
        )
        return service

    def test_handler_execution_success(self, mock_pipeline_service, mocker):
        """Test successful command handling."""
        handler = ExecutePipelineHandler(
            pipeline_service=mock_pipeline_service,
            notification_service=mocker.Mock()
        )

        command = ExecutePipelineCommand(
            pipeline_name="test-pipeline",
            environment="development"
        )

        result = handler.handle(command)

        assert result.success
        assert result.value["execution_id"] == "exec-123"
        assert result.value["status"] == "started"

    def test_handler_pipeline_not_found(self, mocker):
        """Test handler when pipeline not found."""
        mock_service = mocker.Mock()
        mock_service.get_pipeline.return_value = None

        handler = ExecutePipelineHandler(
            pipeline_service=mock_service,
            notification_service=mocker.Mock()
        )

        command = ExecutePipelineCommand(
            pipeline_name="nonexistent-pipeline"
        )

        result = handler.handle(command)

        assert result.is_failure
        assert "Pipeline 'nonexistent-pipeline' not found" in result.error

class TestCliIntegration:
    """Test CLI command integration."""

    @pytest.fixture
    def runner(self):
        """Click test runner."""
        return CliRunner()

    def test_execute_pipeline_cli_success(self, runner, mocker):
        """Test CLI command execution success."""
        # Mock the command bus
        mock_result = FlextResult.ok({
            "execution_id": "exec-123",
            "status": "started",
            "started_at": "2025-01-01T12:00:00Z",
            "pipeline_name": "test-pipeline"
        })
        mocker.patch('cli.command_bus.execute', return_value=mock_result)

        result = runner.invoke(execute_pipeline, [
            'test-pipeline',
            '--env', 'development',
            '--param', 'batch_size=100'
        ])

        assert result.exit_code == 0
        assert "Pipeline execution started successfully" in result.output
        assert "exec-123" in result.output

    def test_execute_pipeline_cli_validation_error(self, runner, mocker):
        """Test CLI command with validation error."""
        mock_result = FlextResult.fail(
            "Invalid environment 'invalid'",
            error_code="VALIDATION_ERROR"
        )
        mocker.patch('cli.command_bus.execute', return_value=mock_result)

        result = runner.invoke(execute_pipeline, [
            'test-pipeline',
            '--env', 'invalid'
        ])

        assert result.exit_code == 1
        assert "Validation error" in result.output

    def test_execute_pipeline_cli_dry_run(self, runner, mocker):
        """Test CLI command in dry run mode."""
        mock_result = FlextResult.ok({
            "dry_run": True,
            "execution_plan": {"steps": []},
            "estimated_duration_minutes": 15,
            "pipeline_name": "test-pipeline"
        })
        mocker.patch('cli.command_bus.execute', return_value=mock_result)

        result = runner.invoke(execute_pipeline, [
            'test-pipeline',
            '--dry-run'
        ])

        assert result.exit_code == 0
        assert "Dry run completed successfully" in result.output
        assert "Estimated duration: 15 minutes" in result.output
```

These examples demonstrate comprehensive CQRS integration patterns for CLI applications, providing structured command processing, robust validation, rich user interfaces, and thorough testing approaches.
