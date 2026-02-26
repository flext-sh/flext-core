from __future__ import annotations

import argparse
import sys

from flext_infra.check.services import PyreflyConfigFixer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("projects", nargs="*")
    _ = parser.add_argument("--dry-run", action="store_true")
    _ = parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    fixer = PyreflyConfigFixer()
    result = fixer.run(
        projects=args.projects,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    if result.is_failure:
        _ = sys.stderr.write(f"{result.error or 'pyrefly config fix failed'}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
