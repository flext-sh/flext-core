"""Tests for flext_infra.basemk.__main__ CLI entry point.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest
from flext_core import r
from flext_infra.basemk.__main__ import _build_config, main
from flext_infra.basemk.generator import FlextInfraBaseMkGenerator


def test_basemk_main_with_no_command() -> None:
    """Test main() with no command prints help and returns 1."""
    with patch("sys.argv", ["basemk"]):
        with patch("flext_infra.output.output"):
            result = main(argv=[])

            assert result == 1


def test_basemk_main_with_generate_command() -> None:
    """Test main() with generate command succeeds."""
    with patch("sys.stdout", new_callable=StringIO):
        result = main(argv=["generate"])

        assert result == 0


def test_basemk_main_with_output_file(tmp_path: Path) -> None:
    """Test main() writes to output file when specified."""
    output_file = tmp_path / "base.mk"

    result = main(argv=["generate", "--output", str(output_file)])

    assert result == 0
    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert len(content) > 0


def test_basemk_main_with_project_name(tmp_path: Path) -> None:
    """Test main() accepts project name override."""
    output_file = tmp_path / "base.mk"

    result = main(
        argv=["generate", "--project-name", "my-project", "--output", str(output_file)]
    )

    assert result == 0
    assert output_file.exists()


def test_basemk_main_with_invalid_command() -> None:
    """Test main() with invalid command raises SystemExit."""
    with patch("flext_infra.output.output"):
        with pytest.raises(SystemExit):
            main(argv=["invalid"])


def test_basemk_main_ensures_structlog_configured() -> None:
    """Test main() ensures structlog is configured."""
    with patch("flext_core.FlextRuntime.ensure_structlog_configured") as mock_ensure:
        with patch("sys.stdout", new_callable=StringIO):
            main(argv=["generate"])

            assert mock_ensure.call_count >= 1


def test_basemk_build_config_with_none() -> None:
    """Test _build_config returns None when project_name is None."""
    result = _build_config(None)

    assert result is None


def test_basemk_build_config_with_project_name() -> None:
    """Test _build_config returns config with project name."""
    result = _build_config("my-project")

    assert result is not None
    assert result.project_name == "my-project"


def test_basemk_main_with_none_argv() -> None:
    """Test main() with None argv uses sys.argv."""
    with patch("sys.argv", ["basemk", "generate"]):
        with patch("sys.stdout", new_callable=StringIO):
            result = main(argv=None)

            assert result == 0


def test_basemk_main_output_to_stdout() -> None:
    """Test main() outputs to stdout when no output file specified."""
    with patch("sys.stdout", new_callable=StringIO):
        result = main(argv=["generate"])

        assert result == 0


def test_basemk_main_with_generation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test main() handles generation failure."""

    def mock_generate(*args: object, **kwargs: object) -> r[str]:
        return r[str].fail("Generation failed")

    monkeypatch.setattr(FlextInfraBaseMkGenerator, "generate", mock_generate)
    result = main(argv=["generate"])
    assert result == 1


def test_basemk_main_calls_sys_exit() -> None:
    """Test main() calls sys.exit."""
    with patch("sys.argv", ["basemk", "generate"]):
        with patch("sys.stdout", new_callable=StringIO):
            with patch("sys.exit") as _mock_exit:
                try:
                    main(argv=["generate"])
                except SystemExit:
                    pass
