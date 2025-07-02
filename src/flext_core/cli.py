"""Command-line interface for FLEXT Core."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from .domain.runtime_types import CorrelationId, EventId, UserId


def info_command(args: Any) -> int:
    """Show FLEXT Core information."""
    return 0


def validate_command(args: Any) -> int:
    """Validate domain objects and types."""
    errors = 0

    # Test type creation
    try:
        from uuid import uuid4

        UserId(uuid4())
        CorrelationId(uuid4())
        EventId(uuid4())
    except Exception:
        errors += 1

    # Test service result
    try:
        from .domain.advanced_types import ServiceResult

        ServiceResult.ok("test data")
    except Exception:
        errors += 1

    if errors == 0:
        return 0
    else:
        return 1


def types_command(args: Any) -> int:
    """Show available domain types."""
    types_info = [
        ("UserId", "User identification value object", "UUID-based"),
        ("CorrelationId", "Request correlation value object", "UUID-based"),
        ("EventId", "Event identification value object", "UUID-based"),
        ("TenantId", "Multi-tenant identification value object", "UUID-based"),
        ("CommandId", "Command identification value object", "UUID-based"),
        ("QueryId", "Query identification value object", "UUID-based"),
    ]

    for _type_name, _description, _example in types_info:
        pass

    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="FLEXT Core - Foundation & Domain Layer CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  flext-core info        # Show core information
  flext-core validate    # Validate core components
  flext-core types       # Show available types
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show FLEXT Core information")
    info_parser.set_defaults(func=info_command)

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate core components")
    validate_parser.set_defaults(func=validate_command)

    # Types command
    types_parser = subparsers.add_parser("types", help="Show available domain types")
    types_parser.set_defaults(func=types_command)

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    if hasattr(args, "func"):
        return args.func(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
