"""Run flext_infra.refactor CLI."""

from __future__ import annotations

from flext_infra.refactor.engine import FlextInfraRefactorEngine


def main() -> None:
    """Module-level CLI entry point."""
    FlextInfraRefactorEngine.main()


if __name__ == "__main__":
    FlextInfraRefactorEngine.main()
