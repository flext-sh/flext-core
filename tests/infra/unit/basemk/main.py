"""Tests for flext_infra.basemk.__main__ CLI entry point.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch

from flext_core import r, t
from flext_infra.basemk.__main__ import _build_config, main
from flext_infra.basemk.generator import FlextInfraBaseMkGenerator
from flext_tests import tm


def test_basemk_main_with_no_command(monkeypatch: MonkeyPatch) -> None:
    """Test main() with no command prints help and returns 1."""
    monkeypatch.setattr(sys, "argv", ["basemk"])
    monkeypatch.setattr(
        "flext_infra.basemk.__main__.output",
        type(
            "FakeOutput",
            (),
            {
                "error": staticmethod(lambda *a, **kw: None),
                "info": staticmethod(lambda *a, **kw: None),
                "warning": staticmethod(lambda *a, **kw: None),
            },
        )(),
    )
    result = main(argv=[])
    tm.that(result, eq=1)


def test_basemk_main_with_generate_command(monkeypatch: MonkeyPatch) -> None:
    """Test main() with generate command succeeds."""
    monkeypatch.setattr(sys, "stdout", StringIO())
    result = main(argv=["generate"])
    tm.that(result, eq=0)


def test_basemk_main_with_output_file(tmp_path: Path) -> None:
    """Test main() writes to output file when specified."""
    output_file = tmp_path / "base.mk"
    result = main(argv=["generate", "--output", str(output_file)])
    tm.that(result, eq=0)
    tm.that(output_file.exists(), eq=True)
    content = output_file.read_text(encoding="utf-8")
    tm.that(len(content) > 0, eq=True)


def test_basemk_main_with_project_name(tmp_path: Path) -> None:
    """Test main() accepts project name override."""
    output_file = tmp_path / "base.mk"
    result = main(
        argv=["generate", "--project-name", "my-project", "--output", str(output_file)],
    )
    tm.that(result, eq=0)
    tm.that(output_file.exists(), eq=True)


def test_basemk_main_with_invalid_command(monkeypatch: MonkeyPatch) -> None:
    """Test main() with invalid command raises SystemExit."""
    monkeypatch.setattr(
        "flext_infra.basemk.__main__.output",
        type(
            "FakeOutput",
            (),
            {
                "error": staticmethod(lambda *a, **kw: None),
                "info": staticmethod(lambda *a, **kw: None),
                "warning": staticmethod(lambda *a, **kw: None),
            },
        )(),
    )
    with pytest.raises(SystemExit):
        main(argv=["invalid"])


def test_basemk_main_ensures_structlog_configured(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test main() ensures structlog is configured."""
    call_count = 0

    def _fake_ensure() -> None:
        nonlocal call_count
        call_count += 1

    monkeypatch.setattr(
        "flext_core.FlextRuntime.ensure_structlog_configured",
        _fake_ensure,
    )
    monkeypatch.setattr(sys, "stdout", StringIO())
    main(argv=["generate"])
    tm.that(call_count >= 1, eq=True)


def test_basemk_build_config_with_none() -> None:
    """Test _build_config returns None when project_name is None."""
    result = _build_config(None)
    tm.that(result is None, eq=True)


def test_basemk_build_config_with_project_name() -> None:
    """Test _build_config returns config with project name."""
    result = _build_config("my-project")
    tm.that(result is not None, eq=True)
    assert result is not None
    tm.that(result.project_name, eq="my-project")


def test_basemk_main_with_none_argv(monkeypatch: MonkeyPatch) -> None:
    """Test main() with None argv uses sys.argv."""
    monkeypatch.setattr(sys, "argv", ["basemk", "generate"])
    monkeypatch.setattr(sys, "stdout", StringIO())
    result = main(argv=None)
    tm.that(result, eq=0)


def test_basemk_main_output_to_stdout(monkeypatch: MonkeyPatch) -> None:
    """Test main() outputs to stdout when no output file specified."""
    monkeypatch.setattr(sys, "stdout", StringIO())
    result = main(argv=["generate"])
    tm.that(result, eq=0)


def test_basemk_main_with_generation_failure(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test main() handles generation failure."""

    def mock_generate(*args: t.Scalar, **kwargs: t.Scalar) -> r[str]:
        return r[str].fail("Generation failed")

    monkeypatch.setattr(FlextInfraBaseMkGenerator, "generate", mock_generate)
    result = main(argv=["generate"])
    tm.that(result, eq=1)


def test_basemk_main_calls_sys_exit(monkeypatch: MonkeyPatch) -> None:
    """Test main() with --help raises SystemExit."""
    monkeypatch.setattr(sys, "stdout", StringIO())
    with pytest.raises(SystemExit) as exc_info:
        main(argv=["--help"])
    tm.that(exc_info.value.code, eq=0)


def test_basemk_main_with_write_failure(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test main() handles write failure gracefully."""
    output_file = tmp_path / "base.mk"

    def mock_write(*args: t.Scalar, **kwargs: t.Scalar) -> r[bool]:
        return r[bool].fail("Write failed")

    monkeypatch.setattr(FlextInfraBaseMkGenerator, "write", mock_write)
    result = main(argv=["generate", "--output", str(output_file)])
    tm.that(result, eq=1)
