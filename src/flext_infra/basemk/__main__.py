"""CLI entry point for base.mk generation utilities."""

from __future__ import annotations

import sys
from pathlib import Path

from flext_infra import m, u
from flext_infra.basemk.engine import FlextInfraBaseMkTemplateEngine
from flext_infra.basemk.generator import FlextInfraBaseMkGenerator


def _build_config(project_name: str | None) -> m.Infra.Basemk.BaseMkConfig | None:
    if project_name is None:
        return None
    return FlextInfraBaseMkTemplateEngine.default_config().model_copy(
        update={"project_name": project_name},
    )


def _main_impl(argv: list[str] | None = None) -> int:
    """Generate base.mk content from templates and write to file or stdout."""
    parser = u.Infra.create_parser(
        "basemk",
        "base.mk generation utilities",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate base.mk content from templates",
    )
    _ = generate_parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root directory (default: cwd)",
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
    generator = FlextInfraBaseMkGenerator()
    config = _build_config(args.project_name)
    generated_result = generator.generate(config)
    if generated_result.is_failure:
        return u.Infra.exit_code(
            generated_result, failure_msg="base.mk generation failed"
        )
    write_result = generator.write(
        generated_result.value,
        output=args.output,
        stream=sys.stdout,
    )
    return u.Infra.exit_code(write_result, failure_msg="base.mk write failed")


def main(argv: list[str] | None = None) -> int:
    """Generate base.mk content from templates and write to file or stdout."""
    return u.Infra.run_cli(_main_impl, argv)


if __name__ == "__main__":
    sys.exit(main())
