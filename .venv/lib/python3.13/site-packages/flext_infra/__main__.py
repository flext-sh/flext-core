"""Process entrypoint for the canonical centralized flext-infra CLI."""

from __future__ import annotations

import sys

from flext_infra.codegen.managed_conflicts_bootstrap import (
    ManagedConflictBootstrapError,
    prepare_managed_conflicts,
)


def _run() -> None:
    """Recover apply metadata before loading the normal facade-backed CLI."""
    try:
        recovered = prepare_managed_conflicts(sys.argv[1:])
    except ManagedConflictBootstrapError as exc:
        sys.stderr.write(f"ERROR: managed conflict bootstrap: {exc}\n")
        raise SystemExit(2) from None
    if recovered is not None:
        sys.stderr.write(
            f"INFO: recovered owner-declared managed conflicts: {recovered}\n"
        )
    from flext_cli import cli
    from flext_infra.cli import main

    cli.exit(main())


if __name__ == "__main__":
    _run()
