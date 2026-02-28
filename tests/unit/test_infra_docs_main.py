"""Tests for documentation CLI entry point.

Tests the main() function and all subcommand handlers (audit, fix, build,
generate, validate) with mocked services and argument parsing.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from flext_core import r
from flext_infra.docs.__main__ import (
    _run_audit,
    _run_build,
    _run_fix,
    _run_generate,
    _run_validate,
    main,
)
from flext_infra.docs.auditor import AuditReport


class TestRunAudit:
    """Tests for _run_audit handler."""

    def test_run_audit_success_no_failures(self) -> None:
        """Test _run_audit returns 0 when audit passes."""
        args = Mock()
        args.root = "."
        args.project = None
        args.projects = None
        args.output_dir = ".reports/docs"
        args.check = "all"
        args.strict = 1

        with patch(
            "flext_infra.docs.__main__.FlextInfraDocAuditor"
        ) as mock_auditor_class:
            mock_auditor = Mock()
            mock_auditor_class.return_value = mock_auditor
            report = AuditReport(
                scope="root", issues=[], checks=["links"], strict=True, passed=True
            )
            mock_auditor.audit.return_value = r[list[AuditReport]].ok([report])

            result = _run_audit(args)

            assert result == 0
            mock_auditor.audit.assert_called_once()

    def test_run_audit_success_with_failures(self) -> None:
        """Test _run_audit returns 1 when audit has failures."""
        args = Mock()
        args.root = "."
        args.project = None
        args.projects = None
        args.output_dir = ".reports/docs"
        args.check = "all"
        args.strict = 1

        with patch(
            "flext_infra.docs.__main__.FlextInfraDocAuditor"
        ) as mock_auditor_class:
            mock_auditor = Mock()
            mock_auditor_class.return_value = mock_auditor
            report = AuditReport(
                scope="root", issues=[], checks=["links"], strict=True, passed=False
            )
            mock_auditor.audit.return_value = r[list[AuditReport]].ok([report])

            result = _run_audit(args)

            assert result == 1

    def test_run_audit_failure(self) -> None:
        """Test _run_audit returns 1 on audit failure."""
        args = Mock()
        args.root = "."
        args.project = None
        args.projects = None
        args.output_dir = ".reports/docs"
        args.check = "all"
        args.strict = 1

        with patch(
            "flext_infra.docs.__main__.FlextInfraDocAuditor"
        ) as mock_auditor_class:
            mock_auditor = Mock()
            mock_auditor_class.return_value = mock_auditor
            mock_auditor.audit.return_value = r[list[AuditReport]].fail("audit error")

            with patch("flext_infra.docs.__main__.output.error"):
                result = _run_audit(args)

            assert result == 1

    def test_run_audit_with_project_filter(self) -> None:
        """Test _run_audit passes project filter to auditor."""
        args = Mock()
        args.root = "."
        args.project = "test-project"
        args.projects = None
        args.output_dir = ".reports/docs"
        args.check = "all"
        args.strict = 1

        with patch(
            "flext_infra.docs.__main__.FlextInfraDocAuditor"
        ) as mock_auditor_class:
            mock_auditor = Mock()
            mock_auditor_class.return_value = mock_auditor
            mock_auditor.audit.return_value = r[list[AuditReport]].ok([])

            _run_audit(args)

            call_kwargs = mock_auditor.audit.call_args[1]
            assert call_kwargs["project"] == "test-project"

    def test_run_audit_with_projects_filter(self) -> None:
        """Test _run_audit passes projects filter to auditor."""
        args = Mock()
        args.root = "."
        args.project = None
        args.projects = "proj1,proj2"
        args.output_dir = ".reports/docs"
        args.check = "all"
        args.strict = 1

        with patch(
            "flext_infra.docs.__main__.FlextInfraDocAuditor"
        ) as mock_auditor_class:
            mock_auditor = Mock()
            mock_auditor_class.return_value = mock_auditor
            mock_auditor.audit.return_value = r[list[AuditReport]].ok([])

            _run_audit(args)

            call_kwargs = mock_auditor.audit.call_args[1]
            assert call_kwargs["projects"] == "proj1,proj2"

    def test_run_audit_with_check_links(self) -> None:
        """Test _run_audit passes check parameter."""
        args = Mock()
        args.root = "."
        args.project = None
        args.projects = None
        args.output_dir = ".reports/docs"
        args.check = "links"
        args.strict = 1

        with patch(
            "flext_infra.docs.__main__.FlextInfraDocAuditor"
        ) as mock_auditor_class:
            mock_auditor = Mock()
            mock_auditor_class.return_value = mock_auditor
            mock_auditor.audit.return_value = r[list[AuditReport]].ok([])

            _run_audit(args)

            call_kwargs = mock_auditor.audit.call_args[1]
            assert call_kwargs["check"] == "links"

    def test_run_audit_strict_mode(self) -> None:
        """Test _run_audit passes strict parameter."""
        args = Mock()
        args.root = "."
        args.project = None
        args.projects = None
        args.output_dir = ".reports/docs"
        args.check = "all"
        args.strict = 0

        with patch(
            "flext_infra.docs.__main__.FlextInfraDocAuditor"
        ) as mock_auditor_class:
            mock_auditor = Mock()
            mock_auditor_class.return_value = mock_auditor
            mock_auditor.audit.return_value = r[list[AuditReport]].ok([])

            _run_audit(args)

            call_kwargs = mock_auditor.audit.call_args[1]
            assert call_kwargs["strict"] is False


class TestRunFix:
    """Tests for _run_fix handler."""

    def test_run_fix_success(self) -> None:
        """Test _run_fix returns 0 on success."""
        args = Mock()
        args.root = "."
        args.project = None
        args.projects = None
        args.output_dir = ".reports/docs"
        args.apply = False

        with patch("flext_infra.docs.__main__.FlextInfraDocFixer") as mock_fixer_class:
            mock_fixer = Mock()
            mock_fixer_class.return_value = mock_fixer
            mock_fixer.fix.return_value = r[list].ok([])

            result = _run_fix(args)

            assert result == 0

    def test_run_fix_failure(self) -> None:
        """Test _run_fix returns 1 on failure."""
        args = Mock()
        args.root = "."
        args.project = None
        args.projects = None
        args.output_dir = ".reports/docs"
        args.apply = False

        with patch("flext_infra.docs.__main__.FlextInfraDocFixer") as mock_fixer_class:
            mock_fixer = Mock()
            mock_fixer_class.return_value = mock_fixer
            mock_fixer.fix.return_value = r[list].fail("fix error")

            with patch("flext_infra.docs.__main__.output.error"):
                result = _run_fix(args)

            assert result == 1

    def test_run_fix_with_apply_flag(self) -> None:
        """Test _run_fix passes apply flag."""
        args = Mock()
        args.root = "."
        args.project = None
        args.projects = None
        args.output_dir = ".reports/docs"
        args.apply = True

        with patch("flext_infra.docs.__main__.FlextInfraDocFixer") as mock_fixer_class:
            mock_fixer = Mock()
            mock_fixer_class.return_value = mock_fixer
            mock_fixer.fix.return_value = r[list].ok([])

            _run_fix(args)

            call_kwargs = mock_fixer.fix.call_args[1]
            assert call_kwargs["apply"] is True


class TestRunBuild:
    """Tests for _run_build handler."""

    def test_run_build_success_no_failures(self) -> None:
        """Test _run_build returns 0 when build passes."""
        args = Mock()
        args.root = "."
        args.project = None
        args.projects = None
        args.output_dir = ".reports/docs"

        with patch(
            "flext_infra.docs.__main__.FlextInfraDocBuilder"
        ) as mock_builder_class:
            mock_builder = Mock()
            mock_builder_class.return_value = mock_builder
            mock_report = Mock()
            mock_report.result = "OK"
            mock_builder.build.return_value = r[list].ok([mock_report])

            result = _run_build(args)

            assert result == 0

    def test_run_build_success_with_failures(self) -> None:
        """Test _run_build returns 1 when build has failures."""
        args = Mock()
        args.root = "."
        args.project = None
        args.projects = None
        args.output_dir = ".reports/docs"

        with patch(
            "flext_infra.docs.__main__.FlextInfraDocBuilder"
        ) as mock_builder_class:
            mock_builder = Mock()
            mock_builder_class.return_value = mock_builder
            mock_report = Mock()
            mock_report.result = "FAIL"
            mock_builder.build.return_value = r[list].ok([mock_report])

            result = _run_build(args)

            assert result == 1

    def test_run_build_failure(self) -> None:
        """Test _run_build returns 1 on build failure."""
        args = Mock()
        args.root = "."
        args.project = None
        args.projects = None
        args.output_dir = ".reports/docs"

        with patch(
            "flext_infra.docs.__main__.FlextInfraDocBuilder"
        ) as mock_builder_class:
            mock_builder = Mock()
            mock_builder_class.return_value = mock_builder
            mock_builder.build.return_value = r[list].fail("build error")

            with patch("flext_infra.docs.__main__.output.error"):
                result = _run_build(args)

            assert result == 1


class TestRunGenerate:
    """Tests for _run_generate handler."""

    def test_run_generate_success(self) -> None:
        """Test _run_generate returns 0 on success."""
        args = Mock()
        args.root = "."
        args.project = None
        args.projects = None
        args.output_dir = ".reports/docs"
        args.apply = False

        with patch(
            "flext_infra.docs.__main__.FlextInfraDocGenerator"
        ) as mock_gen_class:
            mock_gen = Mock()
            mock_gen_class.return_value = mock_gen
            mock_gen.generate.return_value = r[list].ok([])

            result = _run_generate(args)

            assert result == 0

    def test_run_generate_failure(self) -> None:
        """Test _run_generate returns 1 on failure."""
        args = Mock()
        args.root = "."
        args.project = None
        args.projects = None
        args.output_dir = ".reports/docs"
        args.apply = False

        with patch(
            "flext_infra.docs.__main__.FlextInfraDocGenerator"
        ) as mock_gen_class:
            mock_gen = Mock()
            mock_gen_class.return_value = mock_gen
            mock_gen.generate.return_value = r[list].fail("generate error")

            with patch("flext_infra.docs.__main__.output.error"):
                result = _run_generate(args)

            assert result == 1

    def test_run_generate_with_apply_flag(self) -> None:
        """Test _run_generate passes apply flag."""
        args = Mock()
        args.root = "."
        args.project = None
        args.projects = None
        args.output_dir = ".reports/docs"
        args.apply = True

        with patch(
            "flext_infra.docs.__main__.FlextInfraDocGenerator"
        ) as mock_gen_class:
            mock_gen = Mock()
            mock_gen_class.return_value = mock_gen
            mock_gen.generate.return_value = r[list].ok([])

            _run_generate(args)

            call_kwargs = mock_gen.generate.call_args[1]
            assert call_kwargs["apply"] is True


class TestRunValidate:
    """Tests for _run_validate handler."""

    def test_run_validate_success_no_failures(self) -> None:
        """Test _run_validate returns 0 when validation passes."""
        args = Mock()
        args.root = "."
        args.project = None
        args.projects = None
        args.output_dir = ".reports/docs"
        args.check = "all"
        args.apply = False

        with patch(
            "flext_infra.docs.__main__.FlextInfraDocValidator"
        ) as mock_val_class:
            mock_val = Mock()
            mock_val_class.return_value = mock_val
            mock_report = Mock()
            mock_report.result = "OK"
            mock_val.validate.return_value = r[list].ok([mock_report])

            result = _run_validate(args)

            assert result == 0

    def test_run_validate_success_with_failures(self) -> None:
        """Test _run_validate returns 1 when validation has failures."""
        args = Mock()
        args.root = "."
        args.project = None
        args.projects = None
        args.output_dir = ".reports/docs"
        args.check = "all"
        args.apply = False

        with patch(
            "flext_infra.docs.__main__.FlextInfraDocValidator"
        ) as mock_val_class:
            mock_val = Mock()
            mock_val_class.return_value = mock_val
            mock_report = Mock()
            mock_report.result = "FAIL"
            mock_val.validate.return_value = r[list].ok([mock_report])

            result = _run_validate(args)

            assert result == 1

    def test_run_validate_failure(self) -> None:
        """Test _run_validate returns 1 on validation failure."""
        args = Mock()
        args.root = "."
        args.project = None
        args.projects = None
        args.output_dir = ".reports/docs"
        args.check = "all"
        args.apply = False

        with patch(
            "flext_infra.docs.__main__.FlextInfraDocValidator"
        ) as mock_val_class:
            mock_val = Mock()
            mock_val_class.return_value = mock_val
            mock_val.validate.return_value = r[list].fail("validate error")

            with patch("flext_infra.docs.__main__.output.error"):
                result = _run_validate(args)

            assert result == 1

    def test_run_validate_with_check_parameter(self) -> None:
        """Test _run_validate passes check parameter."""
        args = Mock()
        args.root = "."
        args.project = None
        args.projects = None
        args.output_dir = ".reports/docs"
        args.check = "links"
        args.apply = False

        with patch(
            "flext_infra.docs.__main__.FlextInfraDocValidator"
        ) as mock_val_class:
            mock_val = Mock()
            mock_val_class.return_value = mock_val
            mock_val.validate.return_value = r[list].ok([])

            _run_validate(args)

            call_kwargs = mock_val.validate.call_args[1]
            assert call_kwargs["check"] == "links"


class TestMain:
    """Tests for main() entry point."""

    def test_main_with_audit_command(self) -> None:
        """Test main() routes audit command."""
        with patch("sys.argv", ["prog", "audit", "--root", "."]):
            with patch(
                "flext_infra.docs.__main__.FlextInfraDocAuditor"
            ) as mock_auditor_class:
                mock_auditor = Mock()
                mock_auditor_class.return_value = mock_auditor
                mock_auditor.audit.return_value = r[list[AuditReport]].ok([])

                result = main()

                assert result == 0

    def test_main_with_fix_command(self) -> None:
        """Test main() routes fix command."""
        with patch("sys.argv", ["prog", "fix", "--root", "."]):
            with patch(
                "flext_infra.docs.__main__.FlextInfraDocFixer"
            ) as mock_fixer_class:
                mock_fixer = Mock()
                mock_fixer_class.return_value = mock_fixer
                mock_fixer.fix.return_value = r[list].ok([])

                result = main()

                assert result == 0

    def test_main_with_build_command(self) -> None:
        """Test main() routes build command."""
        with patch("sys.argv", ["prog", "build", "--root", "."]):
            with patch(
                "flext_infra.docs.__main__.FlextInfraDocBuilder"
            ) as mock_builder_class:
                mock_builder = Mock()
                mock_builder_class.return_value = mock_builder
                mock_builder.build.return_value = r[list].ok([])

                result = main()

                assert result == 0

    def test_main_with_generate_command(self) -> None:
        """Test main() routes generate command."""
        with patch("sys.argv", ["prog", "generate", "--root", "."]):
            with patch(
                "flext_infra.docs.__main__.FlextInfraDocGenerator"
            ) as mock_gen_class:
                mock_gen = Mock()
                mock_gen_class.return_value = mock_gen
                mock_gen.generate.return_value = r[list].ok([])

                result = main()

                assert result == 0

    def test_main_with_validate_command(self) -> None:
        """Test main() routes validate command."""
        with patch("sys.argv", ["prog", "validate", "--root", "."]):
            with patch(
                "flext_infra.docs.__main__.FlextInfraDocValidator"
            ) as mock_val_class:
                mock_val = Mock()
                mock_val_class.return_value = mock_val
                mock_val.validate.return_value = r[list].ok([])

                result = main()

                assert result == 0

    def test_main_with_no_command_prints_help(self) -> None:
        """Test main() prints help when no command given."""
        with patch("sys.argv", ["prog"]):
            result = main()

            assert result == 1

    def test_main_with_help_flag(self) -> None:
        """Test main() with --help flag."""
        with patch("sys.argv", ["prog", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

    def test_main_with_audit_help(self) -> None:
        """Test main() with audit --help."""
        with patch("sys.argv", ["prog", "audit", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

    def test_main_with_fix_help(self) -> None:
        """Test main() with fix --help."""
        with patch("sys.argv", ["prog", "fix", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

    def test_main_with_build_help(self) -> None:
        """Test main() with build --help."""
        with patch("sys.argv", ["prog", "build", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

    def test_main_with_generate_help(self) -> None:
        """Test main() with generate --help."""
        with patch("sys.argv", ["prog", "generate", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

    def test_main_with_validate_help(self) -> None:
        """Test main() with validate --help."""
        with patch("sys.argv", ["prog", "validate", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

    def test_main_audit_with_custom_root(self) -> None:
        """Test main() audit with custom root path."""
        with patch("sys.argv", ["prog", "audit", "--root", "/custom/path"]):
            with patch(
                "flext_infra.docs.__main__.FlextInfraDocAuditor"
            ) as mock_auditor_class:
                mock_auditor = Mock()
                mock_auditor_class.return_value = mock_auditor
                mock_auditor.audit.return_value = r[list[AuditReport]].ok([])

                main()

                call_args = mock_auditor.audit.call_args[1]
                assert str(call_args["root"]).endswith("custom/path")

    def test_main_audit_with_project_filter(self) -> None:
        """Test main() audit with project filter."""
        with patch("sys.argv", ["prog", "audit", "--project", "test-proj"]):
            with patch(
                "flext_infra.docs.__main__.FlextInfraDocAuditor"
            ) as mock_auditor_class:
                mock_auditor = Mock()
                mock_auditor_class.return_value = mock_auditor
                mock_auditor.audit.return_value = r[list[AuditReport]].ok([])

                main()

                call_kwargs = mock_auditor.audit.call_args[1]
                assert call_kwargs["project"] == "test-proj"

    def test_main_audit_with_strict_flag(self) -> None:
        """Test main() audit with strict flag."""
        with patch("sys.argv", ["prog", "audit", "--strict", "0"]):
            with patch(
                "flext_infra.docs.__main__.FlextInfraDocAuditor"
            ) as mock_auditor_class:
                mock_auditor = Mock()
                mock_auditor_class.return_value = mock_auditor
                mock_auditor.audit.return_value = r[list[AuditReport]].ok([])

                main()

                call_kwargs = mock_auditor.audit.call_args[1]
                assert call_kwargs["strict"] is False

    def test_main_fix_with_apply_flag(self) -> None:
        """Test main() fix with apply flag."""
        with patch("sys.argv", ["prog", "fix", "--apply"]):
            with patch(
                "flext_infra.docs.__main__.FlextInfraDocFixer"
            ) as mock_fixer_class:
                mock_fixer = Mock()
                mock_fixer_class.return_value = mock_fixer
                mock_fixer.fix.return_value = r[list].ok([])

                main()

                call_kwargs = mock_fixer.fix.call_args[1]
                assert call_kwargs["apply"] is True

    def test_main_generate_with_apply_flag(self) -> None:
        """Test main() generate with apply flag."""
        with patch("sys.argv", ["prog", "generate", "--apply"]):
            with patch(
                "flext_infra.docs.__main__.FlextInfraDocGenerator"
            ) as mock_gen_class:
                mock_gen = Mock()
                mock_gen_class.return_value = mock_gen
                mock_gen.generate.return_value = r[list].ok([])

                main()

                call_kwargs = mock_gen.generate.call_args[1]
                assert call_kwargs["apply"] is True

    def test_main_validate_with_apply_flag(self) -> None:
        """Test main() validate with apply flag."""
        with patch("sys.argv", ["prog", "validate", "--apply"]):
            with patch(
                "flext_infra.docs.__main__.FlextInfraDocValidator"
            ) as mock_val_class:
                mock_val = Mock()
                mock_val_class.return_value = mock_val
                mock_val.validate.return_value = r[list].ok([])

                main()

                call_kwargs = mock_val.validate.call_args[1]
                assert call_kwargs["apply"] is True

    def test_main_audit_with_check_parameter(self) -> None:
        """Test main() audit with check parameter."""
        with patch("sys.argv", ["prog", "audit", "--check", "links"]):
            with patch(
                "flext_infra.docs.__main__.FlextInfraDocAuditor"
            ) as mock_auditor_class:
                mock_auditor = Mock()
                mock_auditor_class.return_value = mock_auditor
                mock_auditor.audit.return_value = r[list[AuditReport]].ok([])

                main()

                call_kwargs = mock_auditor.audit.call_args[1]
                assert call_kwargs["check"] == "links"

    def test_main_validate_with_check_parameter(self) -> None:
        """Test main() validate with check parameter."""
        with patch("sys.argv", ["prog", "validate", "--check", "links"]):
            with patch(
                "flext_infra.docs.__main__.FlextInfraDocValidator"
            ) as mock_val_class:
                mock_val = Mock()
                mock_val_class.return_value = mock_val
                mock_val.validate.return_value = r[list].ok([])

                main()

                call_kwargs = mock_val.validate.call_args[1]
                assert call_kwargs["check"] == "links"
