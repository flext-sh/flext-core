"""Tests for release CLI argument parsing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch

from flext_infra.release.__main__ import _parse_args
from flext_tests import tm


class TestReleaseMainParsing:
    """Test argument parsing for release CLI."""

    def test_parse_args_defaults(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog"])
        args = _parse_args()
        tm.that(str(args.root), eq=str(Path()))
        tm.that(args.phase, eq="all")
        tm.that(args.version, eq="")
        tm.that(args.tag, eq="")
        tm.that(args.interactive, eq=1)
        tm.that(args.push, eq=False)
        tm.that(args.dry_run, eq=False)

    def test_parse_args_with_root(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "--root", "/tmp/workspace"])
        args = _parse_args()
        tm.that(str(args.root), eq="/tmp/workspace")

    def test_parse_args_with_phase(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "--phase", "validate"])
        args = _parse_args()
        tm.that(args.phase, eq="validate")

    def test_parse_args_with_version(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "--version", "1.0.0"])
        args = _parse_args()
        tm.that(args.version, eq="1.0.0")

    def test_parse_args_with_tag(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "--tag", "v1.0.0"])
        args = _parse_args()
        tm.that(args.tag, eq="v1.0.0")

    def test_parse_args_with_bump(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "--bump", "minor"])
        args = _parse_args()
        tm.that(args.bump, eq="minor")

    def test_parse_args_with_interactive(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "--interactive", "0"])
        args = _parse_args()
        tm.that(args.interactive, eq=0)

    def test_parse_args_with_push(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "--push"])
        args = _parse_args()
        tm.that(args.push, eq=True)

    def test_parse_args_with_dry_run(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "--dry-run"])
        args = _parse_args()
        tm.that(args.dry_run, eq=True)

    def test_parse_args_with_dev_suffix(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "--dev-suffix"])
        args = _parse_args()
        tm.that(args.dev_suffix, eq=True)

    def test_parse_args_with_next_dev(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "--next-dev"])
        args = _parse_args()
        tm.that(args.next_dev, eq=True)

    def test_parse_args_with_next_bump(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "--next-bump", "patch"])
        args = _parse_args()
        tm.that(args.next_bump, eq="patch")

    def test_parse_args_with_create_branches(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "--create-branches", "0"])
        args = _parse_args()
        tm.that(args.create_branches, eq=0)

    def test_parse_args_with_projects(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "--projects", "proj1", "proj2"])
        args = _parse_args()
        tm.that(args.projects, eq=["proj1", "proj2"])

    def test_parse_args_projects_empty(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "--projects"])
        args = _parse_args()
        tm.that(args.projects, eq=[])
