"""Run flext_infra.refactor CLI."""

from __future__ import annotations

import sys

from flext_infra.refactor.engine import FlextInfraRefactorEngine


def main() -> int:
    """Module-level CLI entry point."""
    FlextInfraRefactorEngine.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
