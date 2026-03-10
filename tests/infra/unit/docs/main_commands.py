"""Tests for documentation CLI — _run_build, _run_generate, _run_validate handlers."""

from __future__ import annotations

import argparse
from collections.abc import Callable

import pytest
from pydantic import BaseModel, Field

from flext_core import r, t
from flext_infra.docs import __main__ as docs_main
from flext_infra.docs.__main__ import _run_build, _run_generate, _run_validate
from flext_infra.docs.builder import FlextInfraDocBuilder
from flext_infra.docs.generator import FlextInfraDocGenerator
from flext_infra.docs.validator import FlextInfraDocValidator
from flext_tests import tm


class _Report(BaseModel):
    result: str = Field(default="OK")


def _cli_args(
    extra_defaults: dict[str, t.ContainerValue | None],
    **overrides: t.ContainerValue,
) -> argparse.Namespace:
    defaults: dict[str, t.ContainerValue | None] = {
        "root": ".",
        "project": None,
        "projects": None,
        "output_dir": ".reports/docs",
        **extra_defaults,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _build_args(**overrides: t.ContainerValue) -> argparse.Namespace:
    return _cli_args({}, **overrides)


def _gen_args(**overrides: t.ContainerValue) -> argparse.Namespace:
    return _cli_args({"apply": False}, **overrides)


def _val_args(**overrides: t.ContainerValue) -> argparse.Namespace:
    return _cli_args({"check": "all", "apply": False}, **overrides)


def _stub_ok(val: list[t.ContainerValue]) -> Callable[..., r[list[t.ContainerValue]]]:
    return lambda *_a, **_kw: r[list[t.ContainerValue]].ok(val)


def _stub_fail(err: str) -> Callable[..., r[list[t.ContainerValue]]]:
    return lambda *_a, **_kw: r[list[t.ContainerValue]].fail(err)


_SILENT_OUTPUT = type("O", (), {"error": staticmethod(lambda *a: None)})()


class TestRunBuild:
    def test_run_build_success_no_failures(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        report = _Report(result="OK")
        monkeypatch.setattr(FlextInfraDocBuilder, "build", _stub_ok([report]))
        tm.that(_run_build(_build_args()), eq=0)

    def test_run_build_success_with_failures(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        report = _Report(result="FAIL")
        monkeypatch.setattr(FlextInfraDocBuilder, "build", _stub_ok([report]))
        tm.that(_run_build(_build_args()), eq=1)

    def test_run_build_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(FlextInfraDocBuilder, "build", _stub_fail("build error"))
        monkeypatch.setattr(docs_main, "output", _SILENT_OUTPUT)
        tm.that(_run_build(_build_args()), eq=1)


class TestRunGenerate:
    def test_run_generate_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(FlextInfraDocGenerator, "generate", _stub_ok([]))
        tm.that(_run_generate(_gen_args()), eq=0)

    def test_run_generate_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            FlextInfraDocGenerator, "generate", _stub_fail("generate error")
        )
        monkeypatch.setattr(docs_main, "output", _SILENT_OUTPUT)
        tm.that(_run_generate(_gen_args()), eq=1)

    def test_run_generate_with_apply_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured_kwargs: dict[str, t.ContainerValue] = {}

        def mock_gen(
            *_a: t.ContainerValue, **kw: t.ContainerValue
        ) -> r[list[t.ContainerValue]]:
            captured_kwargs.update(kw)
            return r[list[t.ContainerValue]].ok([])

        monkeypatch.setattr(FlextInfraDocGenerator, "generate", mock_gen)
        _run_generate(_gen_args(apply=True))
        tm.that(captured_kwargs.get("apply"), eq=True)


class TestRunValidate:
    def test_run_validate_success_no_failures(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        report = _Report(result="OK")
        monkeypatch.setattr(FlextInfraDocValidator, "validate", _stub_ok([report]))
        tm.that(_run_validate(_val_args()), eq=0)

    def test_run_validate_success_with_failures(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        report = _Report(result="FAIL")
        monkeypatch.setattr(FlextInfraDocValidator, "validate", _stub_ok([report]))
        tm.that(_run_validate(_val_args()), eq=1)

    def test_run_validate_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            FlextInfraDocValidator, "validate", _stub_fail("validate error")
        )
        monkeypatch.setattr(docs_main, "output", _SILENT_OUTPUT)
        tm.that(_run_validate(_val_args()), eq=1)

    def test_run_validate_with_check_parameter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured_kwargs: dict[str, t.ContainerValue] = {}

        def mock_val(
            *_a: t.ContainerValue, **kw: t.ContainerValue
        ) -> r[list[t.ContainerValue]]:
            captured_kwargs.update(kw)
            return r[list[t.ContainerValue]].ok([])

        monkeypatch.setattr(FlextInfraDocValidator, "validate", mock_val)
        _run_validate(_val_args(check="links"))
        tm.that(captured_kwargs.get("check"), eq="links")
