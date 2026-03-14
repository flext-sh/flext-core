"""Tests for FlextUtilitiesGenerators to achieve full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
from collections.abc import ItemsView, Iterator, Mapping
from datetime import UTC, datetime, tzinfo
from typing import cast, override

import pytest
from pydantic import BaseModel

from flext_core import m, u

from ._models import _GoodModel

generators_module = importlib.import_module("flext_core._utilities.generators")
runtime_module = importlib.import_module("flext_core.runtime")


class _BrokenMapping(Mapping[str, object]):
    @override
    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(())

    @override
    def __len__(self) -> int:
        return 0

    @override
    def items(self) -> ItemsView[str, object]:
        msg = "boom"
        raise TypeError(msg)


class _BrokenModel:
    def __init__(self) -> None:
        self.model_dump = lambda: (_ for _ in ()).throw(TypeError("dump-failed"))


def test_normalize_context_to_dict_error_paths() -> None:
    with pytest.raises(TypeError, match="Failed to convert Mapping"):
        u._normalize_context_to_dict(_BrokenMapping())
    with pytest.raises(TypeError, match="Failed to dump BaseModel"):
        u._normalize_context_to_dict(
            cast(
                "Mapping[str, object] | BaseModel | None",
                cast("object", _BrokenModel()),
            ),
        )
    with pytest.raises(TypeError, match="Context cannot be None"):
        u._normalize_context_to_dict(None)
    with pytest.raises(TypeError, match="Failed to dump BaseModel int"):
        u._normalize_context_to_dict(
            cast("Mapping[str, object] | BaseModel | None", cast("object", 42)),
        )


def test_enrich_and_ensure_trace_context_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = iter(["trace-x", "span-x", "corr-x"])
    monkeypatch.setattr(
        runtime_module.FlextRuntime,
        "generate_id",
        staticmethod(lambda: next(ids)),
    )
    monkeypatch.setattr(
        runtime_module.FlextRuntime,
        "generate_datetime_utc",
        staticmethod(lambda: datetime(2026, 1, 1, tzinfo=UTC)),
    )
    enriched = u.ensure_trace_context(
        _GoodModel(value=9),
        include_correlation_id=True,
        include_timestamp=True,
    )
    assert enriched["value"] == "9"
    assert enriched["trace_id"] == "trace-x"
    assert enriched["span_id"] == "span-x"
    assert enriched["correlation_id"] == "corr-x"
    assert enriched["timestamp"] == "2026-01-01T00:00:00+00:00"
    existing = {
        "trace_id": "already-trace",
        "span_id": "already-span",
        "correlation_id": "already-corr",
        "timestamp": "already-ts",
    }
    preserved = u.ensure_trace_context(
        existing,
        include_correlation_id=True,
        include_timestamp=True,
    )
    assert preserved == existing


def test_ensure_dict_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    _ = monkeypatch

    class _IterFailMapping(Mapping[str, object]):
        @override
        def __getitem__(self, key: str) -> object:
            raise KeyError(key)

        @override
        def __iter__(self) -> Iterator[str]:
            msg = "iter-fail"
            raise TypeError(msg)

        @override
        def __len__(self) -> int:
            return 1

    raw = {"a": 1}
    assert u.ensure_dict(raw) is raw
    assert u.ensure_dict(_GoodModel(value=5)) == {"value": 5}
    with pytest.raises(TypeError, match=r"Failed to convert Mapping"):
        u.ensure_dict(_IterFailMapping())
    assert u.ensure_dict(None, default={"x": "y"}) == {"x": "y"}
    with pytest.raises(TypeError, match=r"Value cannot be None"):
        u.ensure_dict(None)
    with pytest.raises(TypeError, match=r"Cannot convert int to dict"):
        u.ensure_dict(123)


def test_generate_special_paths_and_dynamic_subclass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = u.generate(kind="id")
    assert isinstance(generated, str)
    assert len(generated) > 0
    fixed_ts = datetime(2026, 1, 2, tzinfo=UTC)

    class _FixedDatetime:
        @staticmethod
        def now(tz: tzinfo | None = None) -> datetime:
            _ = tz
            return fixed_ts

    monkeypatch.setattr(generators_module, "datetime", _FixedDatetime)
    custom = u.generate(
        kind="command",
        include_timestamp=True,
        separator="-",
        parts=("part",),
        length=8,
    )
    assert custom.startswith("cmd-")
    assert "-part-" in custom
    fallback = u.generate(kind="aggregate")
    assert isinstance(fallback, str)
    dynamic = u.create_dynamic_type_subclass(
        "DynCls",
        object,
        m.ConfigMap(root={"value": 10}),
    )
    instance = dynamic()
    assert getattr(instance, "value") == 10


def test_generators_additional_missed_paths() -> None:
    mapping_ctx: Mapping[str, object] = {"a": 1}
    normalized = u._normalize_context_to_dict(mapping_ctx)
    assert normalized == {"a": 1}
    ensured = u.ensure_dict(_GoodModel(value=3))
    assert ensured == {"value": 3}
    generated = u.generate(kind="event", separator="-")
    assert generated.startswith("evt-")


def test_generators_mapping_non_dict_normalization_path() -> None:

    class _SimpleMapping(Mapping[str, object]):
        @override
        def __getitem__(self, key: str) -> object:
            if key == "a":
                return 1
            raise KeyError(key)

        @override
        def __iter__(self) -> Iterator[str]:
            return iter(["a"])

        @override
        def __len__(self) -> int:
            return 1

    normalized = u._normalize_context_to_dict(_SimpleMapping())
    assert normalized == {"a": 1}
