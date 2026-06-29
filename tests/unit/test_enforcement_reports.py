"""Enforcement report model tests."""

from __future__ import annotations

import typing
from typing import Annotated

import pytest

from tests.models import m
from tests.utilities import u


class TestsFlextEnforcementReports:
    @pytest.mark.parametrize(
        "base_cls",
        [
            m.ArbitraryTypesModel,
            m.StrictBoundaryModel,
            m.FlexibleInternalModel,
            m.ImmutableValueModel,
            m.TaggedModel,
            m.ContractModel,
        ],
    )
    def test_base_model_has_enforcement_hook(self, base_cls: type) -> None:
        assert hasattr(base_cls, "__pydantic_init_subclass__")

    def test_empty_report_is_falsy(self) -> None:
        report = m.Report()
        assert not report
        assert report.empty
        assert len(report) == 0

    def test_nonempty_report_is_truthy(self) -> None:
        v = m.Violation(
            qualname="X",
            layer="Model",
            severity="HARD rules",
            message="boom",
        )
        report = m.Report(violations=[v])
        assert report
        assert not report.empty
        assert len(report) == 1
        assert report[0] == "boom"
        assert "boom" in report

    def test_violation_includes_rule_metadata(self) -> None:
        class _M(m.ArbitraryTypesModel):
            data: Annotated[typing.Any, m.Field(description="d")] = None

        report = u.check(_M)
        assert any(
            v.rule_id == "ENFORCE-039" or "no_any" in v.message
            for v in report.violations
        )

    def test_violation_messages_include_rule_identifiers(self) -> None:
        class _M(m.ArbitraryTypesModel):
            data: Annotated[typing.Any, m.Field(description="d")] = None

        report = u.check(_M)
        assert any("[" in v.message and "]" in v.message for v in report.violations)

    def test_merge_reports(self) -> None:
        v = m.Violation(
            qualname="X",
            layer="Model",
            severity="HARD rules",
            message="a",
        )
        w = m.Violation(
            qualname="X",
            layer="Model",
            severity="HARD rules",
            message="b",
        )
        merged = m.Report(violations=[v, w])
        assert len(merged) == 2

    def test_function_local_class_skipped(self) -> None:
        """Classes defined inside functions carry ``<locals>`` in qualname."""

        class _OuterScope:
            def make(self) -> type:
                class Inner:  # nested inside a method → `<locals>` qualname
                    pass

                return Inner

        cls = _OuterScope().make()
        assert "<locals>" in cls.__qualname__
        # check returns empty report for function-local classes
        report = u.check(cls)
        assert all(v.layer != "namespace" for v in report.violations)
