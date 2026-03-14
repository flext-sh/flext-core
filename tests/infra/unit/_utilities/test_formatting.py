from __future__ import annotations

from pathlib import Path

import pytest

from flext_infra import c, u
from flext_infra._utilities.formatting import FlextInfraUtilitiesSubprocess


class TestFormattingRunRuffFix:
    def test_run_ruff_fix_runs_check_and_format(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        target = tmp_path / "sample.py"
        target.write_text("x=1\n", encoding="utf-8")
        calls: list[list[str]] = []

        def _fake_run_checked(
            _self: FlextInfraUtilitiesSubprocess, cmd: list[str]
        ) -> object:
            calls.append(cmd)
            return object()

        monkeypatch.setattr(
            "flext_infra._utilities.formatting.FlextInfraUtilitiesSubprocess.run_checked",
            _fake_run_checked,
        )

        u.Infra.run_ruff_fix(target)

        assert calls[0] == [
            c.Infra.Cli.RUFF,
            c.Infra.Cli.RuffCmd.CHECK,
            "--fix",
            str(target),
        ]
        assert calls[1] == [
            c.Infra.Cli.RUFF,
            c.Infra.Cli.RuffCmd.FORMAT,
            str(target),
        ]

    def test_run_ruff_fix_quiet_adds_flag_and_suppresses_missing_binary(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        target = tmp_path / "sample.py"
        target.write_text("x=1\n", encoding="utf-8")
        calls: list[list[str]] = []

        def _raise_missing(
            _self: FlextInfraUtilitiesSubprocess, cmd: list[str]
        ) -> None:
            calls.append(cmd)
            msg = "ruff not found"
            raise FileNotFoundError(msg)

        monkeypatch.setattr(
            "flext_infra._utilities.formatting.FlextInfraUtilitiesSubprocess.run_checked",
            _raise_missing,
        )

        u.Infra.run_ruff_fix(target, include_format=False, quiet=True)

        assert calls == [
            [
                c.Infra.Cli.RUFF,
                c.Infra.Cli.RuffCmd.CHECK,
                "--fix",
                "--quiet",
                str(target),
            ]
        ]
