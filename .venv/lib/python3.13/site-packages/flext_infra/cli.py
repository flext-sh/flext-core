"""CLI entrypoint for the canonical flext-infra command surface."""

from __future__ import annotations

import sys

from flext_infra import c, t
from flext_infra.services.cli_dispatch import CliDispatchService


class FlextInfraCli(CliDispatchService):
    """Single CLI entry surface for every flext-infra command group."""


def main(args: t.StrSequence | None = None) -> int:
    """Run the canonical flext-infra CLI."""
    cli_args = list(args) if args is not None else sys.argv[1:]
    return FlextInfraCli().main(cli_args)


def docs_main(args: t.StrSequence | None = None) -> int:
    """Run the docs group directly (``flext-docs`` == ``flext-infra docs``)."""
    cli_args = list(args) if args is not None else sys.argv[1:]
    return FlextInfraCli().main([c.Infra.CLI_GROUP_DOCS, *cli_args])


__all__: list[str] = ["FlextInfraCli", "docs_main", "main"]
