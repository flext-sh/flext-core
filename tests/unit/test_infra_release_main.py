"""Tests for flext_infra.release.__main__ CLI entry point.

Tests argument parsing, version resolution, tag resolution, and main flow
with mocked services and sys.argv manipulation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from flext_core import r
from flext_infra.release.__main__ import (
    _parse_args,
    _resolve_tag,
    _resolve_version,
    main,
)


class TestReleaseMainParsing:
    """Test argument parsing for release CLI."""

    def test_parse_args_defaults(self) -> None:
        """Test _parse_args with default values."""
        with patch.object(sys, "argv", ["prog"]):
            args = _parse_args()
            assert args.root == Path()
            assert args.phase == "all"
            assert args.version == ""
            assert args.tag == ""
            assert args.interactive == 1
            assert args.push is False
            assert args.dry_run is False

    def test_parse_args_with_root(self) -> None:
        """Test _parse_args with --root."""
        with patch.object(sys, "argv", ["prog", "--root", "/tmp/workspace"]):
            args = _parse_args()
            assert args.root == Path("/tmp/workspace")

    def test_parse_args_with_phase(self) -> None:
        """Test _parse_args with --phase."""
        with patch.object(sys, "argv", ["prog", "--phase", "validate"]):
            args = _parse_args()
            assert args.phase == "validate"

    def test_parse_args_with_version(self) -> None:
        """Test _parse_args with --version."""
        with patch.object(sys, "argv", ["prog", "--version", "1.0.0"]):
            args = _parse_args()
            assert args.version == "1.0.0"

    def test_parse_args_with_tag(self) -> None:
        """Test _parse_args with --tag."""
        with patch.object(sys, "argv", ["prog", "--tag", "v1.0.0"]):
            args = _parse_args()
            assert args.tag == "v1.0.0"

    def test_parse_args_with_bump(self) -> None:
        """Test _parse_args with --bump."""
        with patch.object(sys, "argv", ["prog", "--bump", "minor"]):
            args = _parse_args()
            assert args.bump == "minor"

    def test_parse_args_with_interactive(self) -> None:
        """Test _parse_args with --interactive."""
        with patch.object(sys, "argv", ["prog", "--interactive", "0"]):
            args = _parse_args()
            assert args.interactive == 0

    def test_parse_args_with_push(self) -> None:
        """Test _parse_args with --push."""
        with patch.object(sys, "argv", ["prog", "--push"]):
            args = _parse_args()
            assert args.push is True

    def test_parse_args_with_dry_run(self) -> None:
        """Test _parse_args with --dry-run."""
        with patch.object(sys, "argv", ["prog", "--dry-run"]):
            args = _parse_args()
            assert args.dry_run is True

    def test_parse_args_with_dev_suffix(self) -> None:
        """Test _parse_args with --dev-suffix."""
        with patch.object(sys, "argv", ["prog", "--dev-suffix"]):
            args = _parse_args()
            assert args.dev_suffix is True

    def test_parse_args_with_next_dev(self) -> None:
        """Test _parse_args with --next-dev."""
        with patch.object(sys, "argv", ["prog", "--next-dev"]):
            args = _parse_args()
            assert args.next_dev is True

    def test_parse_args_with_next_bump(self) -> None:
        """Test _parse_args with --next-bump."""
        with patch.object(sys, "argv", ["prog", "--next-bump", "patch"]):
            args = _parse_args()
            assert args.next_bump == "patch"

    def test_parse_args_with_create_branches(self) -> None:
        """Test _parse_args with --create-branches."""
        with patch.object(sys, "argv", ["prog", "--create-branches", "0"]):
            args = _parse_args()
            assert args.create_branches == 0

    def test_parse_args_with_projects(self) -> None:
        """Test _parse_args with --projects."""
        with patch.object(sys, "argv", ["prog", "--projects", "proj1", "proj2"]):
            args = _parse_args()
            assert args.projects == ["proj1", "proj2"]

    def test_parse_args_projects_empty(self) -> None:
        """Test _parse_args with --projects but no values."""
        with patch.object(sys, "argv", ["prog", "--projects"]):
            args = _parse_args()
            assert args.projects == []


class TestReleaseMainVersionResolution:
    """Test version resolution logic."""

    def test_resolve_version_explicit(self, tmp_path: Path) -> None:
        """Test _resolve_version with explicit --version."""
        args = MagicMock()
        args.version = "1.0.0"
        args.bump = ""
        args.interactive = 1
        with patch(
            "flext_infra.release.__main__.FlextInfraVersioningService"
        ) as mock_vs:
            mock_vs_inst = mock_vs.return_value
            mock_vs_inst.parse_semver.return_value = r[str].ok("1.0.0")
            version = _resolve_version(args, tmp_path)
            assert version == "1.0.0"

    def test_resolve_version_invalid_explicit(self, tmp_path: Path) -> None:
        """Test _resolve_version rejects invalid explicit version."""
        args = MagicMock()
        args.version = "invalid"
        args.bump = ""
        args.interactive = 1
        with patch(
            "flext_infra.release.__main__.FlextInfraVersioningService"
        ) as mock_vs:
            mock_vs_inst = mock_vs.return_value
            mock_vs_inst.parse_semver.return_value = r[str].fail("invalid")
            with pytest.raises(RuntimeError):
                _resolve_version(args, tmp_path)

    def test_resolve_version_from_current(self, tmp_path: Path) -> None:
        """Test _resolve_version reads current version."""
        args = MagicMock()
        args.version = ""
        args.bump = ""
        args.interactive = 0
        with patch(
            "flext_infra.release.__main__.FlextInfraVersioningService"
        ) as mock_vs:
            mock_vs_inst = mock_vs.return_value
            mock_vs_inst.current_workspace_version.return_value = r[str].ok("0.9.0")
            version = _resolve_version(args, tmp_path)
            assert version == "0.9.0"

    def test_resolve_version_current_read_failure(self, tmp_path: Path) -> None:
        """Test _resolve_version handles current version read failure."""
        args = MagicMock()
        args.version = ""
        args.bump = ""
        args.interactive = 1
        with patch(
            "flext_infra.release.__main__.FlextInfraVersioningService"
        ) as mock_vs:
            mock_vs_inst = mock_vs.return_value
            mock_vs_inst.current_workspace_version.return_value = r[str].fail(
                "read error"
            )
            with pytest.raises(RuntimeError):
                _resolve_version(args, tmp_path)

    def test_resolve_version_with_bump(self, tmp_path: Path) -> None:
        """Test _resolve_version bumps version."""
        args = MagicMock()
        args.version = ""
        args.bump = "minor"
        args.interactive = 1
        with patch(
            "flext_infra.release.__main__.FlextInfraVersioningService"
        ) as mock_vs:
            mock_vs_inst = mock_vs.return_value
            mock_vs_inst.current_workspace_version.return_value = r[str].ok("1.0.0")
            mock_vs_inst.bump_version.return_value = r[str].ok("1.1.0")
            version = _resolve_version(args, tmp_path)
            assert version == "1.1.0"

    def test_resolve_version_bump_failure(self, tmp_path: Path) -> None:
        """Test _resolve_version handles bump failure."""
        args = MagicMock()
        args.version = ""
        args.bump = "invalid"
        args.interactive = 1
        with patch(
            "flext_infra.release.__main__.FlextInfraVersioningService"
        ) as mock_vs:
            mock_vs_inst = mock_vs.return_value
            mock_vs_inst.current_workspace_version.return_value = r[str].ok("1.0.0")
            mock_vs_inst.bump_version.return_value = r[str].fail("invalid bump")
            with pytest.raises(RuntimeError):
                _resolve_version(args, tmp_path)

    def test_resolve_version_interactive_input(self, tmp_path: Path) -> None:
        """Test _resolve_version prompts for bump in interactive mode."""
        args = MagicMock()
        args.version = ""
        args.bump = ""
        args.interactive = 1
        with patch(
            "flext_infra.release.__main__.FlextInfraVersioningService"
        ) as mock_vs:
            mock_vs_inst = mock_vs.return_value
            mock_vs_inst.current_workspace_version.return_value = r[str].ok("1.0.0")
            mock_vs_inst.bump_version.return_value = r[str].ok("1.1.0")
            with patch("builtins.input", return_value="minor"):
                version = _resolve_version(args, tmp_path)
                assert version == "1.1.0"

    def test_resolve_version_interactive_invalid_input(self, tmp_path: Path) -> None:
        """Test _resolve_version rejects invalid interactive input."""
        args = MagicMock()
        args.version = ""
        args.bump = ""
        args.interactive = 1
        with patch(
            "flext_infra.release.__main__.FlextInfraVersioningService"
        ) as mock_vs:
            mock_vs_inst = mock_vs.return_value
            mock_vs_inst.current_workspace_version.return_value = r[str].ok("1.0.0")
            with patch("builtins.input", return_value="invalid"):
                with pytest.raises(RuntimeError):
                    _resolve_version(args, tmp_path)

    def test_resolve_version_non_interactive(self, tmp_path: Path) -> None:
        """Test _resolve_version returns current in non-interactive mode."""
        args = MagicMock()
        args.version = ""
        args.bump = ""
        args.interactive = 0
        with patch(
            "flext_infra.release.__main__.FlextInfraVersioningService"
        ) as mock_vs:
            mock_vs_inst = mock_vs.return_value
            mock_vs_inst.current_workspace_version.return_value = r[str].ok("1.0.0")
            version = _resolve_version(args, tmp_path)
            assert version == "1.0.0"


class TestReleaseMainTagResolution:
    """Test tag resolution logic."""

    def test_resolve_tag_explicit(self) -> None:
        """Test _resolve_tag with explicit --tag."""
        args = MagicMock()
        args.tag = "v1.0.0"
        tag = _resolve_tag(args, "1.0.0")
        assert tag == "v1.0.0"

    def test_resolve_tag_invalid_prefix(self) -> None:
        """Test _resolve_tag rejects tag without v prefix."""
        args = MagicMock()
        args.tag = "1.0.0"
        with pytest.raises(RuntimeError):
            _resolve_tag(args, "1.0.0")

    def test_resolve_tag_auto_generated(self) -> None:
        """Test _resolve_tag generates tag from version."""
        args = MagicMock()
        args.tag = ""
        tag = _resolve_tag(args, "1.0.0")
        assert tag == "v1.0.0"


class TestReleaseMainFlow:
    """Test main() orchestration."""

    def test_main_success(self, tmp_path: Path) -> None:
        """Test main() succeeds with valid arguments."""
        with patch.object(
            sys,
            "argv",
            [
                "prog",
                "--root",
                str(tmp_path),
                "--phase",
                "validate",
                "--interactive",
                "0",
            ],
        ):
            with patch("flext_infra.release.__main__.FlextRuntime"):
                with patch(
                    "flext_infra.release.__main__.FlextInfraPathResolver"
                ) as mock_resolver:
                    mock_resolver_inst = mock_resolver.return_value
                    mock_resolver_inst.workspace_root.return_value = r[Path].ok(
                        tmp_path
                    )
                    with patch(
                        "flext_infra.release.__main__.FlextInfraVersioningService"
                    ) as mock_vs:
                        mock_vs_inst = mock_vs.return_value
                        mock_vs_inst.current_workspace_version.return_value = r[str].ok(
                            "1.0.0"
                        )
                        with patch(
                            "flext_infra.release.__main__.FlextInfraReleaseOrchestrator"
                        ) as mock_orch:
                            mock_orch_inst = mock_orch.return_value
                            mock_orch_inst.run_release.return_value = r[bool].ok(True)
                            result = main()
                            assert result == 0

    def test_main_workspace_root_failure(self, tmp_path: Path) -> None:
        """Test main() handles workspace root resolution failure."""
        with patch.object(
            sys,
            "argv",
            [
                "prog",
                "--root",
                str(tmp_path),
                "--phase",
                "validate",
                "--interactive",
                "0",
            ],
        ):
            with patch("flext_infra.release.__main__.FlextRuntime"):
                with patch(
                    "flext_infra.release.__main__.FlextInfraPathResolver"
                ) as mock_resolver:
                    mock_resolver_inst = mock_resolver.return_value
                    mock_resolver_inst.workspace_root.return_value = r[Path].fail(
                        "not found"
                    )
                    with patch("flext_infra.release.__main__.output") as mock_output:
                        result = main()
                        assert result == 1
                        mock_output.error.assert_called_once()

    def test_main_version_resolution_failure(self, tmp_path: Path) -> None:
        """Test main() handles version resolution failure."""
        with patch.object(
            sys,
            "argv",
            [
                "prog",
                "--root",
                str(tmp_path),
                "--phase",
                "version",
                "--version",
                "invalid",
            ],
        ):
            with patch("flext_infra.release.__main__.FlextRuntime"):
                with patch(
                    "flext_infra.release.__main__.FlextInfraPathResolver"
                ) as mock_resolver:
                    mock_resolver_inst = mock_resolver.return_value
                    mock_resolver_inst.workspace_root.return_value = r[Path].ok(
                        tmp_path
                    )
                    with patch(
                        "flext_infra.release.__main__.FlextInfraVersioningService"
                    ) as mock_vs:
                        mock_vs_inst = mock_vs.return_value
                        mock_vs_inst.parse_semver.return_value = r[str].fail("invalid")
                        with patch(
                            "flext_infra.release.__main__.output"
                        ) as mock_output:
                            result = main()
                            assert result == 1
                            mock_output.error.assert_called_once()

    def test_main_release_failure(self, tmp_path: Path) -> None:
        """Test main() handles release orchestration failure."""
        with patch.object(
            sys,
            "argv",
            [
                "prog",
                "--root",
                str(tmp_path),
                "--phase",
                "validate",
                "--interactive",
                "0",
            ],
        ):
            with patch("flext_infra.release.__main__.FlextRuntime"):
                with patch(
                    "flext_infra.release.__main__.FlextInfraPathResolver"
                ) as mock_resolver:
                    mock_resolver_inst = mock_resolver.return_value
                    mock_resolver_inst.workspace_root.return_value = r[Path].ok(
                        tmp_path
                    )
                    with patch(
                        "flext_infra.release.__main__.FlextInfraVersioningService"
                    ) as mock_vs:
                        mock_vs_inst = mock_vs.return_value
                        mock_vs_inst.current_workspace_version.return_value = r[str].ok(
                            "1.0.0"
                        )
                        with patch(
                            "flext_infra.release.__main__.FlextInfraReleaseOrchestrator"
                        ) as mock_orch:
                            mock_orch_inst = mock_orch.return_value
                            mock_orch_inst.run_release.return_value = r[bool].fail(
                                "release failed"
                            )
                            with patch(
                                "flext_infra.release.__main__.output"
                            ) as mock_output:
                                result = main()
                                assert result == 1
                                mock_output.error.assert_called_once()

    def test_main_all_phases(self, tmp_path: Path) -> None:
        """Test main() with --phase all."""
        with patch.object(
            sys,
            "argv",
            ["prog", "--root", str(tmp_path), "--phase", "all", "--interactive", "0"],
        ):
            with patch("flext_infra.release.__main__.FlextRuntime"):
                with patch(
                    "flext_infra.release.__main__.FlextInfraPathResolver"
                ) as mock_resolver:
                    mock_resolver_inst = mock_resolver.return_value
                    mock_resolver_inst.workspace_root.return_value = r[Path].ok(
                        tmp_path
                    )
                    with patch(
                        "flext_infra.release.__main__.FlextInfraVersioningService"
                    ) as mock_vs:
                        mock_vs_inst = mock_vs.return_value
                        mock_vs_inst.current_workspace_version.return_value = r[str].ok(
                            "1.0.0"
                        )
                        with patch(
                            "flext_infra.release.__main__.FlextInfraReleaseOrchestrator"
                        ) as mock_orch:
                            mock_orch_inst = mock_orch.return_value
                            mock_orch_inst.run_release.return_value = r[bool].ok(True)
                            result = main()
                            assert result == 0
                            call_args = mock_orch_inst.run_release.call_args
                            phases = call_args.kwargs["phases"]
                            assert phases == ["validate", "version", "build", "publish"]

    def test_main_with_push(self, tmp_path: Path) -> None:
        """Test main() with --push."""
        with patch.object(
            sys,
            "argv",
            [
                "prog",
                "--root",
                str(tmp_path),
                "--phase",
                "validate",
                "--push",
                "--interactive",
                "0",
            ],
        ):
            with patch("flext_infra.release.__main__.FlextRuntime"):
                with patch(
                    "flext_infra.release.__main__.FlextInfraPathResolver"
                ) as mock_resolver:
                    mock_resolver_inst = mock_resolver.return_value
                    mock_resolver_inst.workspace_root.return_value = r[Path].ok(
                        tmp_path
                    )
                    with patch(
                        "flext_infra.release.__main__.FlextInfraVersioningService"
                    ) as mock_vs:
                        mock_vs_inst = mock_vs.return_value
                        mock_vs_inst.current_workspace_version.return_value = r[str].ok(
                            "1.0.0"
                        )
                        with patch(
                            "flext_infra.release.__main__.FlextInfraReleaseOrchestrator"
                        ) as mock_orch:
                            mock_orch_inst = mock_orch.return_value
                            mock_orch_inst.run_release.return_value = r[bool].ok(True)
                            result = main()
                            assert result == 0
                            call_args = mock_orch_inst.run_release.call_args
                            assert call_args.kwargs["push"] is True

    def test_main_with_dry_run(self, tmp_path: Path) -> None:
        """Test main() with --dry-run."""
        with patch.object(
            sys,
            "argv",
            [
                "prog",
                "--root",
                str(tmp_path),
                "--phase",
                "validate",
                "--dry-run",
                "--interactive",
                "0",
            ],
        ):
            with patch("flext_infra.release.__main__.FlextRuntime"):
                with patch(
                    "flext_infra.release.__main__.FlextInfraPathResolver"
                ) as mock_resolver:
                    mock_resolver_inst = mock_resolver.return_value
                    mock_resolver_inst.workspace_root.return_value = r[Path].ok(
                        tmp_path
                    )
                    with patch(
                        "flext_infra.release.__main__.FlextInfraVersioningService"
                    ) as mock_vs:
                        mock_vs_inst = mock_vs.return_value
                        mock_vs_inst.current_workspace_version.return_value = r[str].ok(
                            "1.0.0"
                        )
                        with patch(
                            "flext_infra.release.__main__.FlextInfraReleaseOrchestrator"
                        ) as mock_orch:
                            mock_orch_inst = mock_orch.return_value
                            mock_orch_inst.run_release.return_value = r[bool].ok(True)
                            result = main()
                            assert result == 0
                            call_args = mock_orch_inst.run_release.call_args
                            assert call_args.kwargs["dry_run"] is True

    def test_main_with_projects(self, tmp_path: Path) -> None:
        """Test main() with --projects."""
        with patch.object(
            sys,
            "argv",
            [
                "prog",
                "--root",
                str(tmp_path),
                "--phase",
                "validate",
                "--projects",
                "proj1",
                "proj2",
            ],
        ):
            with patch("flext_infra.release.__main__.FlextRuntime"):
                with patch(
                    "flext_infra.release.__main__.FlextInfraPathResolver"
                ) as mock_resolver:
                    mock_resolver_inst = mock_resolver.return_value
                    mock_resolver_inst.workspace_root.return_value = r[Path].ok(
                        tmp_path
                    )
                    with patch(
                        "flext_infra.release.__main__.FlextInfraVersioningService"
                    ) as mock_vs:
                        mock_vs_inst = mock_vs.return_value
                        mock_vs_inst.current_workspace_version.return_value = r[str].ok(
                            "1.0.0"
                        )
                        with patch(
                            "flext_infra.release.__main__.FlextInfraReleaseOrchestrator"
                        ) as mock_orch:
                            mock_orch_inst = mock_orch.return_value
                            mock_orch_inst.run_release.return_value = r[bool].ok(True)
                            result = main()
                            assert result == 0
                            call_args = mock_orch_inst.run_release.call_args
                            assert call_args.kwargs["project_names"] == [
                                "proj1",
                                "proj2",
                            ]
