"""CLI entry point for base.mk generation utilities."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flext_core import FlextRuntime

from flext_infra import m, output
from flext_infra.basemk.engine import TemplateEngine
from flext_infra.basemk.generator import BaseMkGenerator


def _build_config(project_name: str | None) -> m.BaseMkConfig | None:
    if project_name is None:
        return None
    return TemplateEngine.default_config().model_copy(
        update={"project_name": project_name},
    )


def main(argv: list[str] | None = None) -> int:
    """Generate base.mk content from templates and write to file or stdout."""
    FlextRuntime.ensure_structlog_configured()
    parser = argparse.ArgumentParser(description="base.mk generation utilities")
    """Generate base.mk content from templates and write to file or stdout."""
    parser = argparse.ArgumentParser(description="base.mk generation utilities")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate base.mk content from templates",
    )
    _ = generate_parser.add_argument(
        "--output",
        type=Path,
        help="Write generated content to file path (defaults to stdout)",
    )
    _ = generate_parser.add_argument(
        "--project-name",
        type=str,
        help="Override project name in generated base.mk",
    )

    args = parser.parse_args(argv)
    if args.command != "generate":
        parser.print_help()
        return 1

    generator = BaseMkGenerator()
    config = _build_config(args.project_name)
    generated_result = generator.generate(config)
    if generated_result.is_failure:
        output.error(generated_result.error or "base.mk generation failed")
        return 1

    write_result = generator.write(
        generated_result.value,
        output=args.output,
        stream=sys.stdout,
    )
    if write_result.is_failure:
        output.error(write_result.error or "base.mk write failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
