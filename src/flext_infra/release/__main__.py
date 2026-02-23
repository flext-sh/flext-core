"""CLI entry point for release orchestration.

Usage:
    python -m flext_infra release --phase validate --dry-run --root .
    python -m flext_infra release --phase version --version 1.0.0 --root .
    python -m flext_infra release --phase all --version 1.0.0 --root . --dry-run

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flext_infra.paths import PathResolver
from flext_infra.release.orchestrator import ReleaseOrchestrator
from flext_infra.versioning import VersioningService


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments for the release runner."""
    parser = argparse.ArgumentParser(description="Release orchestration")
    _ = parser.add_argument("--root", type=Path, default=Path())
    _ = parser.add_argument("--phase", default="all")
    _ = parser.add_argument("--version", default="")
    _ = parser.add_argument("--tag", default="")
    _ = parser.add_argument("--bump", default="")
    _ = parser.add_argument("--interactive", type=int, default=1)
    _ = parser.add_argument("--push", action="store_true", default=False)
    _ = parser.add_argument("--dry-run", action="store_true", default=False)
    _ = parser.add_argument("--dev-suffix", action="store_true", default=False)
    _ = parser.add_argument("--next-dev", action="store_true", default=False)
    _ = parser.add_argument("--next-bump", default="minor")
    _ = parser.add_argument("--create-branches", type=int, default=1)
    _ = parser.add_argument("--projects", nargs="*", default=[])
    return parser.parse_args()


def _resolve_version(args: argparse.Namespace, root: Path) -> str:
    """Determine the target release version based on arguments."""
    versioning = VersioningService()

    if args.version:
        requested = str(args.version)
        parse_result = versioning.parse_semver(requested)
        if parse_result.is_failure:
            msg = parse_result.error or "invalid version"
            raise RuntimeError(msg)
        return requested

    current_result = versioning.current_workspace_version(root)
    if current_result.is_failure:
        msg = current_result.error or "cannot read current version"
        raise RuntimeError(msg)
    current = current_result.value

    if args.bump:
        bump_result = versioning.bump_version(current, args.bump)
        if bump_result.is_failure:
            msg = bump_result.error or "bump failed"
            raise RuntimeError(msg)
        return bump_result.value

    if args.interactive != 1:
        return current

    print("Select version bump type: [major|minor|patch]")
    bump = input("bump> ").strip().lower()
    if bump not in {"major", "minor", "patch"}:
        msg = "invalid bump type"
        raise RuntimeError(msg)
    bump_result = versioning.bump_version(current, bump)
    if bump_result.is_failure:
        msg = bump_result.error or "bump failed"
        raise RuntimeError(msg)
    return bump_result.value


def _resolve_tag(args: argparse.Namespace, version: str) -> str:
    """Determine the Git tag for the release."""
    if args.tag:
        requested = str(args.tag)
        if not requested.startswith("v"):
            msg = "tag must start with v"
            raise RuntimeError(msg)
        return requested
    return f"v{version}"


def main() -> int:
    """Orchestrate the release process through configured phases."""
    args = _parse_args()

    resolver = PathResolver()
    root_result = resolver.workspace_root(args.root)
    if root_result.is_failure:
        print(f"Error: {root_result.error}", file=sys.stderr)
        return 1
    root = root_result.value

    phases = (
        ["validate", "version", "build", "publish"]
        if args.phase == "all"
        else [part.strip() for part in args.phase.split(",") if part.strip()]
    )

    needs_version = bool({"version", "build", "publish"} & set(phases))
    if needs_version:
        try:
            version = _resolve_version(args, root)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    else:
        version = args.version or "0.0.0"

    tag = _resolve_tag(args, version)

    service = ReleaseOrchestrator()
    result = service.run_release(
        root=root,
        version=version,
        tag=tag,
        phases=phases,
        project_names=args.projects or None,
        dry_run=args.dry_run,
        push=args.push,
        dev_suffix=args.dev_suffix,
        create_branches=args.create_branches == 1,
        next_dev=args.next_dev,
        next_bump=args.next_bump,
    )

    if result.is_failure:
        print(f"Error: {result.error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
