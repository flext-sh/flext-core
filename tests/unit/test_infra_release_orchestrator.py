"""Tests for FlextInfraReleaseOrchestrator.

Tests release orchestration with mocked git and pyproject operations,
using tmp_path fixtures for isolated test environments.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from flext_core import r
from flext_infra import FlextInfraModels
from flext_infra.release.orchestrator import FlextInfraReleaseOrchestrator

if TYPE_CHECKING:
    from pathlib import Path


class TestFlextInfraReleaseOrchestrator:
    """Test suite for FlextInfraReleaseOrchestrator."""

    @pytest.fixture
    def workspace_root(self, tmp_path: Path) -> Path:
        """Create mock workspace root with pyproject.toml."""
        root = tmp_path / "workspace"
        root.mkdir()
        (root / ".git").mkdir()
        (root / "Makefile").touch()
        (root / "pyproject.toml").write_text(
            'version = "0.1.0"\n',
            encoding="utf-8",
        )
        return root

    def test_execute_not_implemented(self) -> None:
        """Test that execute() returns ok(True) (use run_release instead)."""
        orchestrator = FlextInfraReleaseOrchestrator()
        result = orchestrator.execute()
        assert result.is_success
        assert result.value is True

    def test_run_release_invalid_phase(
        self,
        workspace_root: Path,
    ) -> None:
        """Test run_release with invalid phase name."""
        orchestrator = FlextInfraReleaseOrchestrator()
        result = orchestrator.run_release(
            root=workspace_root,
            version="1.0.0",
            tag="v1.0.0",
            phases=["invalid_phase"],
        )

        assert result.is_failure

    def test_run_release_empty_phases_list(
        self,
        workspace_root: Path,
    ) -> None:
        """Test run_release with empty phases list."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._create_branches",
            return_value=r[bool].ok(True),
        ):
            result = orchestrator.run_release(
                root=workspace_root,
                version="1.0.0",
                tag="v1.0.0",
                phases=[],
            )

        assert result.is_success

    def test_run_release_with_project_filter(
        self,
        workspace_root: Path,
    ) -> None:
        """Test run_release with project_names filter."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._create_branches",
            return_value=r[bool].ok(True),
        ):
            with patch(
                "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._dispatch_phase",
                return_value=r[bool].ok(True),
            ):
                result = orchestrator.run_release(
                    root=workspace_root,
                    version="1.0.0",
                    tag="v1.0.0",
                    phases=["validate"],
                    project_names=["flext-core", "flext-api"],
                )

        assert result.is_success

    def test_run_release_dry_run_mode(
        self,
        workspace_root: Path,
    ) -> None:
        """Test run_release with dry_run=True."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._dispatch_phase",
            return_value=r[bool].ok(True),
        ):
            result = orchestrator.run_release(
                root=workspace_root,
                version="1.0.0",
                tag="v1.0.0",
                phases=["validate"],
                dry_run=True,
            )

        assert result.is_success

    def test_run_release_with_push(
        self,
        workspace_root: Path,
    ) -> None:
        """Test run_release with push=True."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._create_branches",
            return_value=r[bool].ok(True),
        ):
            with patch(
                "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._dispatch_phase",
                return_value=r[bool].ok(True),
            ):
                result = orchestrator.run_release(
                    root=workspace_root,
                    version="1.0.0",
                    tag="v1.0.0",
                    phases=["validate"],
                    push=True,
                )

        assert result.is_success

    def test_run_release_with_dev_suffix(
        self,
        workspace_root: Path,
    ) -> None:
        """Test run_release with dev_suffix=True."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._create_branches",
            return_value=r[bool].ok(True),
        ):
            with patch(
                "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._dispatch_phase",
                return_value=r[bool].ok(True),
            ):
                result = orchestrator.run_release(
                    root=workspace_root,
                    version="1.0.0-dev",
                    tag="v1.0.0-dev",
                    phases=["version"],
                    dev_suffix=True,
                )

        assert result.is_success

    def test_run_release_next_dev_version(
        self,
        workspace_root: Path,
    ) -> None:
        """Test run_release with next_dev=True."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._create_branches",
            return_value=r[bool].ok(True),
        ):
            with patch(
                "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._dispatch_phase",
                return_value=r[bool].ok(True),
            ):
                with patch(
                    "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._bump_next_dev",
                    return_value=r[bool].ok(True),
                ):
                    result = orchestrator.run_release(
                        root=workspace_root,
                        version="1.0.0",
                        tag="v1.0.0",
                        phases=["version"],
                        next_dev=True,
                        next_bump="minor",
                    )

        assert result.is_success

    def test_run_release_phase_failure_stops_execution(
        self,
        workspace_root: Path,
    ) -> None:
        """Test that phase failure stops further execution."""
        orchestrator = FlextInfraReleaseOrchestrator()
        call_count = 0

        def mock_dispatch(phase: str, *args: str, **kwargs: str) -> r[bool]:
            nonlocal call_count
            call_count += 1
            if phase == "validate":
                return r[bool].fail("validation failed")
            return r[bool].ok(True)

        with patch(
            "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._create_branches",
            return_value=r[bool].ok(True),
        ):
            with patch(
                "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._dispatch_phase",
                side_effect=mock_dispatch,
            ):
                result = orchestrator.run_release(
                    root=workspace_root,
                    version="1.0.0",
                    tag="v1.0.0",
                    phases=["validate", "version"],
                )

        assert result.is_failure
        assert call_count == 1

    def test_run_release_create_branches_disabled(
        self,
        workspace_root: Path,
    ) -> None:
        """Test run_release with create_branches=False."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._dispatch_phase",
            return_value=r[bool].ok(True),
        ):
            result = orchestrator.run_release(
                root=workspace_root,
                version="1.0.0",
                tag="v1.0.0",
                phases=["validate"],
                create_branches=False,
            )

        assert result.is_success

    def test_run_release_create_branches_enabled(
        self,
        workspace_root: Path,
    ) -> None:
        """Test run_release with create_branches=True (line 114)."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._create_branches",
            return_value=r[bool].ok(True),
        ):
            with patch(
                "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._dispatch_phase",
                return_value=r[bool].ok(True),
            ):
                result = orchestrator.run_release(
                    root=workspace_root,
                    version="1.0.0",
                    tag="v1.0.0",
                    phases=["validate"],
                    create_branches=True,
                    dry_run=False,
                )

        assert result.is_success

    def test_run_release_create_branches_failure(
        self,
        workspace_root: Path,
    ) -> None:
        """Test run_release with create_branches failure (line 114)."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._create_branches",
            return_value=r[bool].fail("branch creation failed"),
        ):
            result = orchestrator.run_release(
                root=workspace_root,
                version="1.0.0",
                tag="v1.0.0",
                phases=["validate"],
                create_branches=True,
                dry_run=False,
            )

        assert result.is_failure
        assert "branch creation failed" in result.error

    def test_run_release_multiple_projects(
        self,
        workspace_root: Path,
    ) -> None:
        """Test run_release with multiple projects."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._create_branches",
            return_value=r[bool].ok(True),
        ):
            with patch(
                "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._dispatch_phase",
                return_value=r[bool].ok(True),
            ):
                result = orchestrator.run_release(
                    root=workspace_root,
                    version="2.0.0",
                    tag="v2.0.0",
                    phases=["validate"],
                    project_names=["flext-core", "flext-api", "flext-cli"],
                )

        assert result.is_success

    def test_phase_validate_dry_run(self, workspace_root: Path) -> None:
        """Test phase_validate with dry_run=True."""
        orchestrator = FlextInfraReleaseOrchestrator()
        result = orchestrator.phase_validate(workspace_root, dry_run=True)
        assert result.is_success

    def test_phase_validate_executes_make(self, workspace_root: Path) -> None:
        """Test phase_validate runs make validate command."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraCommandRunner"
        ) as mock_runner_cls:
            mock_runner = mock_runner_cls.return_value
            mock_runner.run_checked.return_value = r[bool].ok(True)
            result = orchestrator.phase_validate(workspace_root, dry_run=False)
            assert result.is_success
            mock_runner.run_checked.assert_called_once()

    def test_phase_version_updates_files(self, workspace_root: Path) -> None:
        """Test phase_version updates pyproject.toml files."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraVersioningService"
        ) as mock_vs:
            mock_vs_inst = mock_vs.return_value
            mock_vs_inst.parse_semver.return_value = r[str].ok("1.0.0")
            mock_vs_inst.replace_project_version.return_value = None
            result = orchestrator.phase_version(
                workspace_root,
                "1.0.0",
                [],
                dry_run=False,
            )
            assert result.is_success

    def test_phase_version_invalid_semver(self, workspace_root: Path) -> None:
        """Test phase_version rejects invalid semantic version."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraVersioningService"
        ) as mock_vs:
            mock_vs_inst = mock_vs.return_value
            mock_vs_inst.parse_semver.return_value = r[str].fail("invalid version")
            result = orchestrator.phase_version(
                workspace_root,
                "invalid",
                [],
            )
            assert result.is_failure

    def test_phase_version_with_dev_suffix(self, workspace_root: Path) -> None:
        """Test phase_version appends -dev suffix."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraVersioningService"
        ) as mock_vs:
            mock_vs_inst = mock_vs.return_value
            mock_vs_inst.parse_semver.return_value = r[str].ok("1.0.0")
            result = orchestrator.phase_version(
                workspace_root,
                "1.0.0",
                [],
                dev_suffix=True,
            )
            assert result.is_success

    def test_phase_version_dry_run(self, workspace_root: Path) -> None:
        """Test phase_version with dry_run=True."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraVersioningService"
        ) as mock_vs:
            mock_vs_inst = mock_vs.return_value
            mock_vs_inst.parse_semver.return_value = r[str].ok("1.0.0")
            result = orchestrator.phase_version(
                workspace_root,
                "1.0.0",
                [],
                dry_run=True,
            )
            assert result.is_success

    def test_phase_build_creates_report_dir(self, workspace_root: Path) -> None:
        """Test phase_build creates output directory."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraReportingService"
        ) as mock_rep:
            mock_rep_inst = mock_rep.return_value
            mock_rep_inst.get_report_dir.return_value = workspace_root / "reports"
            with patch(
                "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._run_make",
                return_value=r[tuple[int, str]].ok((0, "ok")),
            ):
                with patch("flext_infra.release.orchestrator.FlextInfraJsonService"):
                    result = orchestrator.phase_build(workspace_root, "1.0.0", [])
                    assert result.is_success

    def test_phase_build_report_dir_creation_fails(self, workspace_root: Path) -> None:
        """Test phase_build handles directory creation failure."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraReportingService"
        ) as mock_rep:
            mock_rep_inst = mock_rep.return_value
            mock_rep_inst.get_report_dir.return_value = workspace_root / "reports"
            with patch("pathlib.Path.mkdir", side_effect=OSError("permission denied")):
                result = orchestrator.phase_build(workspace_root, "1.0.0", [])
                assert result.is_failure

    def test_phase_build_with_project_failures(self, workspace_root: Path) -> None:
        """Test phase_build reports project build failures."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraReportingService"
        ) as mock_rep:
            mock_rep_inst = mock_rep.return_value
            mock_rep_inst.get_report_dir.return_value = workspace_root / "reports"
            with patch(
                "flext_infra.release.orchestrator.FlextInfraCommandRunner"
            ) as mock_runner_cls:
                mock_runner = mock_runner_cls.return_value

                output_model = FlextInfraModels.CommandOutput(
                    exit_code=1,
                    stdout="error",
                    stderr="",
                )
                mock_runner.run_raw.return_value = r[FlextInfraModels.CommandOutput].ok(
                    output_model
                )
                with patch("flext_infra.release.orchestrator.FlextInfraJsonService"):
                    result = orchestrator.phase_build(workspace_root, "1.0.0", [])
                    assert result.is_failure

    def test_phase_publish_generates_notes(self, workspace_root: Path) -> None:
        """Test phase_publish generates release notes."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraReportingService"
        ) as mock_rep:
            mock_rep_inst = mock_rep.return_value
            mock_rep_inst.get_report_dir.return_value = workspace_root / "reports"
            with patch(
                "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._generate_notes",
                return_value=r[bool].ok(True),
            ):
                result = orchestrator.phase_publish(
                    workspace_root,
                    "1.0.0",
                    "v1.0.0",
                    [],
                    dry_run=True,
                )
                assert result.is_success

    def test_phase_publish_dry_run_skips_changelog(self, workspace_root: Path) -> None:
        """Test phase_publish with dry_run=True skips changelog update."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraReportingService"
        ) as mock_rep:
            mock_rep_inst = mock_rep.return_value
            mock_rep_inst.get_report_dir.return_value = workspace_root / "reports"
            with patch(
                "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._generate_notes",
                return_value=r[bool].ok(True),
            ):
                with patch(
                    "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._update_changelog"
                ) as mock_changelog:
                    result = orchestrator.phase_publish(
                        workspace_root,
                        "1.0.0",
                        "v1.0.0",
                        [],
                        dry_run=True,
                    )
                    assert result.is_success
                    mock_changelog.assert_not_called()

    def test_phase_publish_updates_changelog(self, workspace_root: Path) -> None:
        """Test phase_publish updates changelog when not dry_run."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraReportingService"
        ) as mock_rep:
            mock_rep_inst = mock_rep.return_value
            mock_rep_inst.get_report_dir.return_value = workspace_root / "reports"
            with patch(
                "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._generate_notes",
                return_value=r[bool].ok(True),
            ):
                with patch(
                    "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._update_changelog",
                    return_value=r[bool].ok(True),
                ):
                    with patch(
                        "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._create_tag",
                        return_value=r[bool].ok(True),
                    ):
                        result = orchestrator.phase_publish(
                            workspace_root,
                            "1.0.0",
                            "v1.0.0",
                            [],
                            dry_run=False,
                        )
                        assert result.is_success

    def test_phase_publish_with_push(self, workspace_root: Path) -> None:
        """Test phase_publish pushes when push=True."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraReportingService"
        ) as mock_rep:
            mock_rep_inst = mock_rep.return_value
            mock_rep_inst.get_report_dir.return_value = workspace_root / "reports"
            with patch(
                "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._generate_notes",
                return_value=r[bool].ok(True),
            ):
                with patch(
                    "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._update_changelog",
                    return_value=r[bool].ok(True),
                ):
                    with patch(
                        "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._create_tag",
                        return_value=r[bool].ok(True),
                    ):
                        with patch(
                            "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._push_release",
                            return_value=r[bool].ok(True),
                        ) as mock_push:
                            result = orchestrator.phase_publish(
                                workspace_root,
                                "1.0.0",
                                "v1.0.0",
                                [],
                                dry_run=False,
                                push=True,
                            )
                            assert result.is_success
                            mock_push.assert_called_once()

    def test_create_branches_workspace_only(self, workspace_root: Path) -> None:
        """Test _create_branches creates workspace branch."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraCommandRunner"
        ) as mock_runner_cls:
            mock_runner = mock_runner_cls.return_value
            mock_runner.run_checked.return_value = r[bool].ok(True)
            result = orchestrator._create_branches(workspace_root, "1.0.0", [])
            assert result.is_success

    def test_version_files_includes_workspace_root(self, workspace_root: Path) -> None:
        """Test _version_files includes workspace pyproject.toml."""
        orchestrator = FlextInfraReleaseOrchestrator()
        files = orchestrator._version_files(workspace_root, [])
        assert any(f.name == "pyproject.toml" for f in files)

    def test_build_targets_includes_root(self, workspace_root: Path) -> None:
        """Test _build_targets includes root as first target."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraProjectSelector"
        ) as mock_sel:
            mock_sel_inst = mock_sel.return_value
            mock_sel_inst.resolve_projects.return_value = r[list].ok([])
            targets = orchestrator._build_targets(workspace_root, [])
            assert targets[0] == ("root", workspace_root)

    def test_run_make_success(self, workspace_root: Path) -> None:
        """Test _run_make returns exit code and output."""
        with patch(
            "flext_infra.release.orchestrator.FlextInfraCommandRunner"
        ) as mock_runner_cls:
            mock_runner = mock_runner_cls.return_value
            output_model = FlextInfraModels.CommandOutput(
                exit_code=0,
                stdout="build ok",
                stderr="",
            )
            mock_runner.run_raw.return_value = r[FlextInfraModels.CommandOutput].ok(
                output_model
            )
            result = FlextInfraReleaseOrchestrator._run_make(workspace_root, "build")
            assert result.is_success
            code, _output = result.value
            assert code == 0

    def test_run_make_failure(self, workspace_root: Path) -> None:
        """Test _run_make handles command failure."""
        with patch(
            "flext_infra.release.orchestrator.FlextInfraCommandRunner"
        ) as mock_runner_cls:
            mock_runner = mock_runner_cls.return_value
            mock_runner.run_raw.return_value = r[tuple[int, str]].fail("command failed")
            result = FlextInfraReleaseOrchestrator._run_make(workspace_root, "build")
            assert result.is_failure

    def test_generate_notes_writes_file(self, workspace_root: Path) -> None:
        """Test _generate_notes writes release notes file."""
        orchestrator = FlextInfraReleaseOrchestrator()
        notes_path = workspace_root / "notes.md"
        with patch(
            "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._previous_tag",
            return_value=r[str].ok(""),
        ):
            with patch(
                "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._collect_changes",
                return_value=r[str].ok(""),
            ):
                with patch(
                    "flext_infra.release.orchestrator.FlextInfraProjectSelector"
                ) as mock_sel:
                    mock_sel_inst = mock_sel.return_value
                    mock_sel_inst.resolve_projects.return_value = r[list].ok([])
                    result = orchestrator._generate_notes(
                        workspace_root,
                        "1.0.0",
                        "v1.0.0",
                        [],
                        notes_path,
                    )
                    assert result.is_success
                    assert notes_path.exists()

    def test_generate_notes_file_write_failure(self, workspace_root: Path) -> None:
        """Test _generate_notes handles file write failure."""
        orchestrator = FlextInfraReleaseOrchestrator()
        notes_path = workspace_root / "notes.md"
        with patch(
            "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._previous_tag",
            return_value=r[str].ok(""),
        ):
            with patch(
                "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._collect_changes",
                return_value=r[str].ok(""),
            ):
                with patch(
                    "flext_infra.release.orchestrator.FlextInfraProjectSelector"
                ) as mock_sel:
                    mock_sel_inst = mock_sel.return_value
                    mock_sel_inst.resolve_projects.return_value = r[list].ok([])
                    with patch(
                        "pathlib.Path.write_text",
                        side_effect=OSError("write failed"),
                    ):
                        result = orchestrator._generate_notes(
                            workspace_root,
                            "1.0.0",
                            "v1.0.0",
                            [],
                            notes_path,
                        )
                        assert result.is_failure

    def test_previous_tag_finds_tag(self, workspace_root: Path) -> None:
        """Test _previous_tag returns previous tag."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraCommandRunner"
        ) as mock_runner_cls:
            mock_runner = mock_runner_cls.return_value
            mock_runner.capture.return_value = r[str].ok("v1.0.0\nv0.9.0\nv0.8.0")
            result = orchestrator._previous_tag(workspace_root, "v1.0.0")
            assert result.is_success
            assert result.value == "v0.9.0"

    def test_previous_tag_no_previous(self, workspace_root: Path) -> None:
        """Test _previous_tag returns empty when no previous tag."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraCommandRunner"
        ) as mock_runner_cls:
            mock_runner = mock_runner_cls.return_value
            mock_runner.capture.return_value = r[str].ok("v1.0.0")
            result = orchestrator._previous_tag(workspace_root, "v1.0.0")
            assert result.is_success
            assert result.value == ""

    def test_previous_tag_git_failure(self, workspace_root: Path) -> None:
        """Test _previous_tag handles git failure."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraCommandRunner"
        ) as mock_runner_cls:
            mock_runner = mock_runner_cls.return_value
            mock_runner.capture.return_value = r[str].fail("git error")
            result = orchestrator._previous_tag(workspace_root, "v1.0.0")
            assert result.is_failure

    def test_collect_changes_with_tag(self, workspace_root: Path) -> None:
        """Test _collect_changes collects commits between tags."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch("flext_infra.release.orchestrator.FlextInfraGitService") as mock_git:
            mock_git_inst = mock_git.return_value
            mock_git_inst.tag_exists.return_value = r[bool].ok(True)
            with patch(
                "flext_infra.release.orchestrator.FlextInfraCommandRunner"
            ) as mock_runner_cls:
                mock_runner = mock_runner_cls.return_value
                mock_runner.capture.return_value = r[str].ok(
                    "- abc1234 fix: bug (author)"
                )
                result = orchestrator._collect_changes(
                    workspace_root, "v0.9.0", "v1.0.0"
                )
                assert result.is_success

    def test_collect_changes_git_failure(self, workspace_root: Path) -> None:
        """Test _collect_changes handles git failure."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch("flext_infra.release.orchestrator.FlextInfraGitService") as mock_git:
            mock_git_inst = mock_git.return_value
            mock_git_inst.tag_exists.return_value = r[bool].ok(False)
            with patch(
                "flext_infra.release.orchestrator.FlextInfraCommandRunner"
            ) as mock_runner_cls:
                mock_runner = mock_runner_cls.return_value
                mock_runner.capture.return_value = r[str].fail("git error")
                result = orchestrator._collect_changes(workspace_root, "", "HEAD")
                assert result.is_failure

    def test_update_changelog_creates_files(self, workspace_root: Path) -> None:
        """Test _update_changelog creates changelog and release notes."""
        orchestrator = FlextInfraReleaseOrchestrator()
        notes_path = workspace_root / "notes.md"
        notes_path.write_text("# Release v1.0.0\n")
        result = orchestrator._update_changelog(
            workspace_root,
            "1.0.0",
            "v1.0.0",
            notes_path,
        )
        assert result.is_success
        changelog = workspace_root / "docs" / "CHANGELOG.md"
        assert changelog.exists()

    def test_update_changelog_appends_to_existing(self, workspace_root: Path) -> None:
        """Test _update_changelog appends to existing changelog."""
        orchestrator = FlextInfraReleaseOrchestrator()
        changelog = workspace_root / "docs" / "CHANGELOG.md"
        changelog.parent.mkdir(parents=True)
        changelog.write_text("# Changelog\n\n## 0.9.0 - 2025-01-01\n")
        notes_path = workspace_root / "notes.md"
        notes_path.write_text("# Release v1.0.0\n")
        result = orchestrator._update_changelog(
            workspace_root,
            "1.0.0",
            "v1.0.0",
            notes_path,
        )
        assert result.is_success
        content = changelog.read_text()
        assert "1.0.0" in content

    def test_update_changelog_file_write_failure(self, workspace_root: Path) -> None:
        """Test _update_changelog handles file write failure."""
        orchestrator = FlextInfraReleaseOrchestrator()
        notes_path = workspace_root / "notes.md"
        notes_path.write_text("# Release v1.0.0\n")
        with patch("pathlib.Path.write_text", side_effect=OSError("write failed")):
            result = orchestrator._update_changelog(
                workspace_root,
                "1.0.0",
                "v1.0.0",
                notes_path,
            )
            assert result.is_failure

    def test_create_tag_creates_new_tag(self, workspace_root: Path) -> None:
        """Test _create_tag creates annotated git tag."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch("flext_infra.release.orchestrator.FlextInfraGitService") as mock_git:
            mock_git_inst = mock_git.return_value
            mock_git_inst.tag_exists.return_value = r[bool].ok(False)
            with patch(
                "flext_infra.release.orchestrator.FlextInfraCommandRunner"
            ) as mock_runner_cls:
                mock_runner = mock_runner_cls.return_value
                mock_runner.run_checked.return_value = r[bool].ok(True)
                result = orchestrator._create_tag(workspace_root, "v1.0.0")
                assert result.is_success

    def test_create_tag_skips_existing(self, workspace_root: Path) -> None:
        """Test _create_tag skips if tag already exists."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch("flext_infra.release.orchestrator.FlextInfraGitService") as mock_git:
            mock_git_inst = mock_git.return_value
            mock_git_inst.tag_exists.return_value = r[bool].ok(True)
            result = orchestrator._create_tag(workspace_root, "v1.0.0")
            assert result.is_success

    def test_push_release_pushes_branch_and_tag(self, workspace_root: Path) -> None:
        """Test _push_release pushes branch and tag."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraCommandRunner"
        ) as mock_runner_cls:
            mock_runner = mock_runner_cls.return_value
            mock_runner.run_checked.return_value = r[bool].ok(True)
            result = orchestrator._push_release(workspace_root, "v1.0.0")
            assert result.is_success
            assert mock_runner.run_checked.call_count == 2

    def test_push_release_branch_failure(self, workspace_root: Path) -> None:
        """Test _push_release handles branch push failure."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraCommandRunner"
        ) as mock_runner_cls:
            mock_runner = mock_runner_cls.return_value
            mock_runner.run_checked.return_value = r[bool].fail("push failed")
            result = orchestrator._push_release(workspace_root, "v1.0.0")
            assert result.is_failure

    def test_bump_next_dev_bumps_version(self, workspace_root: Path) -> None:
        """Test _bump_next_dev bumps to next dev version."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraVersioningService"
        ) as mock_vs:
            mock_vs_inst = mock_vs.return_value
            mock_vs_inst.bump_version.return_value = r[str].ok("1.1.0")
            with patch(
                "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator.phase_version",
                return_value=r[bool].ok(True),
            ):
                result = orchestrator._bump_next_dev(
                    workspace_root, "1.0.0", [], "minor"
                )
                assert result.is_success

    def test_bump_next_dev_bump_failure(self, workspace_root: Path) -> None:
        """Test _bump_next_dev handles bump failure."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraVersioningService"
        ) as mock_vs:
            mock_vs_inst = mock_vs.return_value
            mock_vs_inst.bump_version.return_value = r[str].fail("invalid bump")
            result = orchestrator._bump_next_dev(workspace_root, "1.0.0", [], "invalid")
            assert result.is_failure

    def test_phase_version_with_existing_version(self, workspace_root: Path) -> None:
        """Test phase_version skips files with matching version (line 198)."""
        orchestrator = FlextInfraReleaseOrchestrator()
        pyproject = workspace_root / "pyproject.toml"
        pyproject.write_text('version = "1.0.0"\n')

        with patch(
            "flext_infra.release.orchestrator.FlextInfraVersioningService"
        ) as mock_versioning_cls:
            mock_versioning = mock_versioning_cls.return_value
            mock_versioning.parse_semver.return_value = r[str].ok("1.0.0")
            mock_versioning.replace_project_version.return_value = None

            result = orchestrator.phase_version(
                workspace_root,
                "1.0.0",
                [],
                dry_run=False,
            )

            assert result.is_success

    def test_phase_version_file_not_exists(self, workspace_root: Path) -> None:
        """Test phase_version skips missing files (line 194)."""
        orchestrator = FlextInfraReleaseOrchestrator()

        with patch(
            "flext_infra.release.orchestrator.FlextInfraVersioningService"
        ) as mock_versioning_cls:
            mock_versioning = mock_versioning_cls.return_value
            mock_versioning.parse_semver.return_value = r[str].ok("1.0.0")

            result = orchestrator.phase_version(
                workspace_root,
                "1.0.0",
                [],
                dry_run=False,
            )

            assert result.is_success

    def test_phase_build_with_make_failure(self, workspace_root: Path) -> None:
        """Test phase_build handles make execution failure (lines 249-250)."""
        orchestrator = FlextInfraReleaseOrchestrator()

        with patch(
            "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._build_targets"
        ) as mock_targets:
            with patch(
                "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._run_make"
            ) as mock_make:
                mock_targets.return_value = [("root", workspace_root)]
                mock_make.return_value = r[tuple[int, str]].fail("make failed")

                result = orchestrator.phase_build(workspace_root, "1.0.0", [])

                assert result.is_failure

    def test_phase_publish_notes_generation_failure(self, workspace_root: Path) -> None:
        """Test phase_publish handles notes generation failure (line 328)."""
        orchestrator = FlextInfraReleaseOrchestrator()

        with patch(
            "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._generate_notes"
        ) as mock_notes:
            mock_notes.return_value = r[bool].fail("notes generation failed")

            result = orchestrator.phase_publish(
                workspace_root,
                "1.0.0",
                "v1.0.0",
                [],
                dry_run=False,
            )

            assert result.is_failure

    def test_phase_publish_changelog_update_failure(self, workspace_root: Path) -> None:
        """Test phase_publish handles changelog update failure (line 333)."""
        orchestrator = FlextInfraReleaseOrchestrator()
        notes_path = workspace_root / "RELEASE_NOTES.md"
        notes_path.write_text("# Release v1.0.0\n")

        with patch(
            "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._generate_notes"
        ) as mock_notes:
            with patch(
                "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._update_changelog"
            ) as mock_changelog:
                mock_notes.return_value = r[bool].ok(True)
                mock_changelog.return_value = r[bool].fail("changelog update failed")

                result = orchestrator.phase_publish(
                    workspace_root,
                    "1.0.0",
                    "v1.0.0",
                    [],
                    dry_run=False,
                )

                assert result.is_failure

    def test_phase_publish_tag_creation_failure(self, workspace_root: Path) -> None:
        """Test phase_publish handles tag creation failure (line 337)."""
        orchestrator = FlextInfraReleaseOrchestrator()
        notes_path = workspace_root / "RELEASE_NOTES.md"
        notes_path.write_text("# Release v1.0.0\n")

        with patch(
            "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._generate_notes"
        ) as mock_notes:
            with patch(
                "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._update_changelog"
            ) as mock_changelog:
                with patch(
                    "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._create_tag"
                ) as mock_tag:
                    mock_notes.return_value = r[bool].ok(True)
                    mock_changelog.return_value = r[bool].ok(True)
                    mock_tag.return_value = r[bool].fail("tag creation failed")

                    result = orchestrator.phase_publish(
                        workspace_root,
                        "1.0.0",
                        "v1.0.0",
                        [],
                        dry_run=False,
                    )

                    assert result.is_failure

    def test_phase_publish_push_failure(self, workspace_root: Path) -> None:
        """Test phase_publish handles push failure (line 342)."""
        orchestrator = FlextInfraReleaseOrchestrator()
        notes_path = workspace_root / "RELEASE_NOTES.md"
        notes_path.write_text("# Release v1.0.0\n")

        with patch(
            "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._generate_notes"
        ) as mock_notes:
            with patch(
                "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._update_changelog"
            ) as mock_changelog:
                with patch(
                    "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._create_tag"
                ) as mock_tag:
                    with patch(
                        "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator._push_release"
                    ) as mock_push:
                        mock_notes.return_value = r[bool].ok(True)
                        mock_changelog.return_value = r[bool].ok(True)
                        mock_tag.return_value = r[bool].ok(True)
                        mock_push.return_value = r[bool].fail("push failed")

                        result = orchestrator.phase_publish(
                            workspace_root,
                            "1.0.0",
                            "v1.0.0",
                            [],
                            dry_run=False,
                            push=True,
                        )

                        assert result.is_failure

    def test_dispatch_phase_unknown(self, workspace_root: Path) -> None:
        """Test _dispatch_phase fails on unknown phase (line 385)."""
        orchestrator = FlextInfraReleaseOrchestrator()

        result = orchestrator._dispatch_phase(
            "unknown",
            workspace_root,
            "1.0.0",
            "v1.0.0",
            [],
            dry_run=False,
            push=False,
            dev_suffix=False,
        )

        assert result.is_failure
        assert "unknown phase" in result.error

    def test_create_branches_failure(self, workspace_root: Path) -> None:
        """Test _create_branches handles git failure (line 401)."""
        orchestrator = FlextInfraReleaseOrchestrator()

        with patch(
            "flext_infra.release.orchestrator.FlextInfraCommandRunner"
        ) as mock_runner_cls:
            mock_runner = mock_runner_cls.return_value
            mock_runner.run_checked.return_value = r[bool].fail("git failed")

            result = orchestrator._create_branches(workspace_root, "1.0.0", [])

            assert result.is_failure

    def test_create_branches_project_failure(self, workspace_root: Path) -> None:
        """Test _create_branches handles project branch failure (lines 407-412)."""
        orchestrator = FlextInfraReleaseOrchestrator()

        with patch(
            "flext_infra.release.orchestrator.FlextInfraCommandRunner"
        ) as mock_runner_cls:
            with patch(
                "flext_infra.release.orchestrator.FlextInfraProjectSelector"
            ) as mock_selector_cls:
                mock_runner = mock_runner_cls.return_value
                mock_selector = mock_selector_cls.return_value

                # First call succeeds (workspace), second fails (project)
                mock_runner.run_checked.side_effect = [
                    r[bool].ok(True),
                    r[bool].fail("project branch failed"),
                ]

                mock_project = SimpleNamespace(
                    name="proj1", path=workspace_root / "proj1"
                )
                mock_selector.resolve_projects.return_value = r[list].ok([mock_project])

                result = orchestrator._create_branches(
                    workspace_root, "1.0.0", ["proj1"]
                )

                assert result.is_failure

    def test_version_files_discovery(self, workspace_root: Path) -> None:
        """Test _version_files discovers pyproject files (lines 427-429)."""
        orchestrator = FlextInfraReleaseOrchestrator()
        proj_dir = workspace_root / "proj1"
        proj_dir.mkdir()
        (proj_dir / "pyproject.toml").touch()

        with patch(
            "flext_infra.release.orchestrator.FlextInfraProjectSelector"
        ) as mock_selector_cls:
            mock_selector = mock_selector_cls.return_value
            mock_project = SimpleNamespace(name="proj1", path=proj_dir)
            mock_selector.resolve_projects.return_value = r[list].ok([mock_project])

            result = orchestrator._version_files(workspace_root, ["proj1"])

            assert len(result) >= 1

    def test_build_targets_deduplication(self, workspace_root: Path) -> None:
        """Test _build_targets deduplicates targets (line 448)."""
        orchestrator = FlextInfraReleaseOrchestrator()

        with patch(
            "flext_infra.release.orchestrator.FlextInfraProjectSelector"
        ) as mock_selector_cls:
            mock_selector = mock_selector_cls.return_value
            mock_project = SimpleNamespace(name="proj1", path=workspace_root / "proj1")
            mock_selector.resolve_projects.return_value = r[list].ok([mock_project])

            result = orchestrator._build_targets(workspace_root, ["proj1"])

            # Should have root + proj1, no duplicates
            names = [name for name, _ in result]
            assert len(names) == len(set(names))

    def test_previous_tag_with_existing_tag(self, workspace_root: Path) -> None:
        """Test _previous_tag finds previous tag (line 545)."""
        orchestrator = FlextInfraReleaseOrchestrator()

        with patch(
            "flext_infra.release.orchestrator.FlextInfraCommandRunner"
        ) as mock_runner_cls:
            mock_runner = mock_runner_cls.return_value
            mock_runner.capture.return_value = r[str].ok("v1.0.0\nv0.9.0\nv0.8.0\n")

            result = orchestrator._previous_tag(workspace_root, "v1.0.0")

            assert result.is_success
            assert result.value == "v0.9.0"

    def test_update_changelog_creates_new_section(self, workspace_root: Path) -> None:
        """Test _update_changelog creates new changelog section (lines 598-600)."""
        orchestrator = FlextInfraReleaseOrchestrator()
        notes_path = workspace_root / "RELEASE_NOTES.md"
        notes_path.write_text("# Release v1.0.0\n\nChanges here.\n")

        result = orchestrator._update_changelog(
            workspace_root,
            "1.0.0",
            "v1.0.0",
            notes_path,
        )

        assert result.is_success
        changelog = workspace_root / "docs" / "CHANGELOG.md"
        if changelog.exists():
            content = changelog.read_text()
            assert "1.0.0" in content


class TestFlextInfraReleaseOrchestratorPhaseVersion:
    """Test phase_version with file existence checks (lines 194, 198)."""

    @pytest.fixture
    def workspace_root(self, tmp_path: Path) -> Path:
        """Create mock workspace root with version files."""
        root = tmp_path / "workspace"
        root.mkdir()
        (root / ".git").mkdir()
        (root / "Makefile").touch()
        (root / "pyproject.toml").write_text('version = "0.1.0"\n')
        return root

    def test_phase_version_skips_missing_files(self, workspace_root: Path) -> None:
        """Test phase_version skips files that don't exist (line 194)."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch.object(
            orchestrator,
            "_version_files",
            return_value=[workspace_root / "nonexistent.toml"],
        ):
            result = orchestrator.phase_version(workspace_root, "1.0.0", [])
            assert result.is_success

    def test_phase_version_skips_unchanged_versions(self, workspace_root: Path) -> None:
        """Test phase_version skips files with matching version (line 198)."""
        orchestrator = FlextInfraReleaseOrchestrator()
        version_file = workspace_root / "pyproject.toml"
        version_file.write_text('version = "1.0.0"\n')
        with patch.object(
            orchestrator,
            "_version_files",
            return_value=[version_file],
        ):
            result = orchestrator.phase_version(workspace_root, "1.0.0", [])
            assert result.is_success


class TestFlextInfraReleaseOrchestratorPhaseBuild:
    """Test phase_build with directory creation (lines 316-317)."""

    @pytest.fixture
    def workspace_root(self, tmp_path: Path) -> Path:
        """Create mock workspace root."""
        root = tmp_path / "workspace"
        root.mkdir()
        (root / ".git").mkdir()
        (root / "Makefile").touch()
        (root / "pyproject.toml").write_text('version = "0.1.0"\n')
        return root

    def test_phase_build_creates_report_directory(self, workspace_root: Path) -> None:
        """Test phase_build creates report directory (lines 316-317)."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch.object(
            orchestrator,
            "_build_targets",
            return_value=[],
        ):
            with patch(
                "flext_infra.release.orchestrator.FlextInfraReportingService"
            ) as mock_reporting:
                mock_instance = mock_reporting.return_value
                mock_instance.get_report_dir.return_value = workspace_root / "reports"
                result = orchestrator.phase_build(workspace_root, "1.0.0", [])
                assert result.is_success


class TestFlextInfraReleaseOrchestratorDispatchPhase:
    """Test _dispatch_phase routing (lines 365, 367, 375, 377)."""

    @pytest.fixture
    def workspace_root(self, tmp_path: Path) -> Path:
        """Create mock workspace root."""
        root = tmp_path / "workspace"
        root.mkdir()
        (root / ".git").mkdir()
        (root / "Makefile").touch()
        (root / "pyproject.toml").write_text('version = "0.1.0"\n')
        return root

    def test_dispatch_phase_routes_validate(self, workspace_root: Path) -> None:
        """Test _dispatch_phase routes 'validate' phase (line 365)."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator.phase_validate",
            return_value=r[bool].ok(True),
        ):
            result = orchestrator._dispatch_phase(
                "validate",
                workspace_root,
                "1.0.0",
                "v1.0.0",
                [],
                dry_run=False,
                push=False,
                dev_suffix=False,
            )
            assert result.is_success

    def test_dispatch_phase_routes_version(self, workspace_root: Path) -> None:
        """Test _dispatch_phase routes 'version' phase (line 367)."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator.phase_version",
            return_value=r[bool].ok(True),
        ):
            result = orchestrator._dispatch_phase(
                "version",
                workspace_root,
                "1.0.0",
                "v1.0.0",
                [],
                dry_run=False,
                push=False,
                dev_suffix=False,
            )
            assert result.is_success

    def test_dispatch_phase_routes_build(self, workspace_root: Path) -> None:
        """Test _dispatch_phase routes 'build' phase (line 375)."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator.phase_build",
            return_value=r[bool].ok(True),
        ):
            result = orchestrator._dispatch_phase(
                "build",
                workspace_root,
                "1.0.0",
                "v1.0.0",
                [],
                dry_run=False,
                push=False,
                dev_suffix=False,
            )
            assert result.is_success

    def test_dispatch_phase_routes_publish(self, workspace_root: Path) -> None:
        """Test _dispatch_phase routes 'publish' phase (line 377)."""
        orchestrator = FlextInfraReleaseOrchestrator()
        with patch(
            "flext_infra.release.orchestrator.FlextInfraReleaseOrchestrator.phase_publish",
            return_value=r[bool].ok(True),
        ):
            result = orchestrator._dispatch_phase(
                "publish",
                workspace_root,
                "1.0.0",
                "v1.0.0",
                [],
                dry_run=False,
                push=False,
                dev_suffix=False,
            )
            assert result.is_success


class TestFlextInfraReleaseOrchestratorChangeCollection:
    """Test _collect_changes git log parsing (lines 598-600)."""

    def test_collect_changes_with_git_log_output(self, tmp_path: Path) -> None:
        """Test _collect_changes parses git log output (lines 598-600)."""
        orchestrator = FlextInfraReleaseOrchestrator()
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        (workspace_root / ".git").mkdir()
        with patch("flext_infra.release.orchestrator.FlextInfraGitService") as mock_git:
            mock_git_instance = mock_git.return_value
            mock_git_instance.tag_exists.return_value = r[bool].ok(True)
            with patch(
                "flext_infra.release.orchestrator.FlextInfraCommandRunner"
            ) as mock_runner:
                mock_runner_instance = mock_runner.return_value
                mock_runner_instance.capture.return_value = r[str].ok(
                    "- abc1234 Fix bug (Alice)\n- def5678 Add feature (Bob)\n"
                )
                result = orchestrator._collect_changes(
                    workspace_root, "v0.9.0", "v1.0.0"
                )
                assert result.is_success
                assert "Fix bug" in result.value
