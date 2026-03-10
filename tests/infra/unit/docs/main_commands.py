"""Tests for documentation CLI — _run_build, _run_generate, _run_validate handlers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from flext_core import r
from flext_infra.docs import __main__ as docs_main
from flext_infra.docs.__main__ import _run_build, _run_generate, _run_validate
from flext_infra.docs.builder import FlextInfraDocBuilder
from flext_infra.docs.generator import FlextInfraDocGenerator
from flext_infra.docs.validator import FlextInfraDocValidator
from flext_tests import tm


def _build_args(**overrides: object) -> argparse.Namespace:
    """Build build CLI args namespace."""
    defaults = {
        "root": ".",
        "project": None,
        "projects": None,
        "output_dir": ".reports/docs",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _gen_args(**overrides: object) -> argparse.Namespace:
    """Build generate CLI args namespace."""
    defaults = {
        "root": ".",
        "project": None,
        "projects": None,
        "output_dir": ".reports/docs",
        "apply": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _val_args(**overrides: object) -> argparse.Namespace:
    """Build validate CLI args namespace."""
    defaults = {
        "root": ".",
        "project": None,
        "projects": None,
        "output_dir": ".reports/docs",
        "check": "all",
        "apply": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestRunBuild:
    """Tests for _run_build handler."""

    def test_run_build_success_no_failures(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test _run_build returns 0 when build passes."""
        report = SimpleNamespace(result="OK")

        def mock_build(*a: object, **kw: object) -> r[list[object]]:
            return r[list[object]].ok([report])

        monkeypatch.setattr(FlextInfraDocBuilder, "build", mock_build)
        tm.that(_run_build(_build_args()), eq=0)

    def test_run_build_success_with_failures(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test _run_build returns 1 when build has failures."""
        report = SimpleNamespace(result="FAIL")

        def mock_build(*a: object, **kw: object) -> r[list[object]]:
            return r[list[object]].ok([report])

        monkeypatch.setattr(FlextInfraDocBuilder, "build", mock_build)
        tm.that(_run_build(_build_args()), eq=1)

    def test_run_build_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test _run_build returns 1 on build failure."""

        def mock_build(*a: object, **kw: object) -> r[list[object]]:
            return r[list[object]].fail("build error")

        monkeypatch.setattr(FlextInfraDocBuilder, "build", mock_build)
        monkeypatch.setattr(
            docs_main,
            "output",
            type("O", (), {"error": staticmethod(lambda *a: None)})(),
        )
        tm.that(_run_build(_build_args()), eq=1)


class TestRunGenerate:
    """Tests for _run_generate handler."""

    def test_run_generate_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test _run_generate returns 0 on success."""

        def mock_gen(*a: object, **kw: object) -> r[list[object]]:
            return r[list[object]].ok([])

        monkeypatch.setattr(FlextInfraDocGenerator, "generate", mock_gen)
        tm.that(_run_generate(_gen_args()), eq=0)

    def test_run_generate_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test _run_generate returns 1 on failure."""

        def mock_gen(*a: object, **kw: object) -> r[list[object]]:
            return r[list[object]].fail("generate error")

        monkeypatch.setattr(FlextInfraDocGenerator, "generate", mock_gen)
        monkeypatch.setattr(
            docs_main,
            "output",
            type("O", (), {"error": staticmethod(lambda *a: None)})(),
        )
        tm.that(_run_generate(_gen_args()), eq=1)

    def test_run_generate_with_apply_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test _run_generate passes apply flag."""
        captured_kwargs: dict[str, object] = {}

        def mock_gen(*a: object, **kw: object) -> r[list[object]]:
            captured_kwargs.update(kw)
            return r[list[object]].ok([])

        monkeypatch.setattr(FlextInfraDocGenerator, "generate", mock_gen)
        _run_generate(_gen_args(apply=True))
        tm.that(captured_kwargs.get("apply"), eq=True)


class TestRunValidate:
    """Tests for _run_validate handler."""

    def test_run_validate_success_no_failures(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test _run_validate returns 0 when validation passes."""
        report = SimpleNamespace(result="OK")

        def mock_val(*a: object, **kw: object) -> r[list[object]]:
            return r[list[object]].ok([report])

        monkeypatch.setattr(FlextInfraDocValidator, "validate", mock_val)
        tm.that(_run_validate(_val_args()), eq=0)

    def test_run_validate_success_with_failures(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test _run_validate returns 1 when validation has failures."""
        report = SimpleNamespace(result="FAIL")

        def mock_val(*a: object, **kw: object) -> r[list[object]]:
            return r[list[object]].ok([report])

        monkeypatch.setattr(FlextInfraDocValidator, "validate", mock_val)
        tm.that(_run_validate(_val_args()), eq=1)

    def test_run_validate_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test _run_validate returns 1 on validation failure."""

        def mock_val(*a: object, **kw: object) -> r[list[object]]:
            return r[list[object]].fail("validate error")

        monkeypatch.setattr(FlextInfraDocValidator, "validate", mock_val)
        monkeypatch.setattr(
            docs_main,
            "output",
            type("O", (), {"error": staticmethod(lambda *a: None)})(),
        )
        tm.that(_run_validate(_val_args()), eq=1)

    def test_run_validate_with_check_parameter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test _run_validate passes check parameter."""
        captured_kwargs: dict[str, object] = {}

        def mock_val(*a: object, **kw: object) -> r[list[object]]:
            captured_kwargs.update(kw)
            return r[list[object]].ok([])

        monkeypatch.setattr(FlextInfraDocValidator, "validate", mock_val)
        _run_validate(_val_args(check="links"))
        tm.that(captured_kwargs.get("check"), eq="links")
