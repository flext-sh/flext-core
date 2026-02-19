from __future__ import annotations

from collections.abc import ItemsView, Iterator, Mapping
from datetime import UTC, datetime

import importlib
import pytest
from pydantic import BaseModel

core = importlib.import_module("flext_core")
m = core.m
t = core.t
u = core.u

generators_module = importlib.import_module("flext_core._utilities.generators")


class _BrokenMapping(Mapping[str, t.GeneralValueType]):
    def __getitem__(self, key: str) -> t.GeneralValueType:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self) -> ItemsView[str, t.GeneralValueType]:
        msg = "boom"
        raise TypeError(msg)


class _GoodModel(BaseModel):
    value: int = 7


class _BrokenModel(BaseModel):
    value: int = 1

    def model_dump(self, **kwargs: object) -> dict[str, t.GeneralValueType]:
        _ = kwargs
        msg = "dump-failed"
        raise TypeError(msg)


def test_normalize_context_to_dict_error_paths() -> None:
    with pytest.raises(TypeError, match="Failed to convert Mapping"):
        u.Generators._normalize_context_to_dict(_BrokenMapping())

    with pytest.raises(TypeError, match="Failed to dump BaseModel"):
        u.Generators._normalize_context_to_dict(_BrokenModel())

    with pytest.raises(TypeError, match="Context cannot be None"):
        u.Generators._normalize_context_to_dict(None)

    with pytest.raises(TypeError, match="Context must be dict, Mapping, or BaseModel"):
        u.Generators._normalize_context_to_dict(42)


def test_enrich_and_ensure_trace_context_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = iter(["trace-x", "span-x", "corr-x"])
    monkeypatch.setattr(
        generators_module.FlextUtilitiesGenerators,
        "_generate_id",
        staticmethod(lambda: next(ids)),
    )
    monkeypatch.setattr(
        generators_module.FlextUtilitiesGenerators,
        "generate_iso_timestamp",
        staticmethod(lambda: "2026-01-01T00:00:00+00:00"),
    )

    enriched = u.Generators.ensure_trace_context(
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
    preserved = u.Generators.ensure_trace_context(
        existing,
        include_correlation_id=True,
        include_timestamp=True,
    )
    assert preserved == existing


def test_ensure_dict_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = {"a": 1}
    assert u.Generators.ensure_dict(raw) is raw

    monkeypatch.setattr(
        "flext_core.runtime.FlextRuntime.normalize_to_general_value",
        staticmethod(lambda value: "not-a-dict"),
    )
    assert u.Generators.ensure_dict(_GoodModel(value=5)) == {}

    with pytest.raises(TypeError, match="Failed to convert Mapping"):
        u.Generators.ensure_dict(_BrokenMapping())

    assert u.Generators.ensure_dict(None, default={"x": "y"}) == {"x": "y"}

    with pytest.raises(TypeError, match="Value cannot be None"):
        u.Generators.ensure_dict(None)

    with pytest.raises(TypeError, match="Cannot convert int to dict"):
        u.Generators.ensure_dict(123)


def test_generate_special_paths_and_dynamic_subclass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = u.Generators.generate(kind="id")
    assert isinstance(generated, str)
    assert len(generated) > 0

    fixed_ts = datetime(2026, 1, 2, tzinfo=UTC)

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz: object | None = None) -> datetime:
            _ = tz
            return fixed_ts

    monkeypatch.setattr("flext_core._utilities.generators.datetime", _FixedDatetime)
    custom = u.Generators.generate(
        kind="command",
        include_timestamp=True,
        separator="-",
        parts=("part",),
        length=8,
    )
    assert custom.startswith("cmd-")
    assert "-part-" in custom

    fallback = u.Generators.generate(kind="aggregate")
    assert isinstance(fallback, str)

    dynamic = u.Generators.create_dynamic_type_subclass(
        "DynCls",
        object,
        m.ConfigMap(root={"value": 10}).root,
    )
    instance = dynamic()
    assert getattr(instance, "value") == 10


def test_generators_additional_missed_paths() -> None:
    mapping_ctx: Mapping[str, t.GeneralValueType] = {"a": 1}
    normalized = u.Generators._normalize_context_to_dict(mapping_ctx)
    assert normalized == {"a": 1}

    ensured = u.Generators.ensure_dict(_GoodModel(value=3))
    assert ensured == {"value": 3}

    generated = u.Generators.generate(kind="event", separator="-")
    assert generated.startswith("evt-")


def test_generators_mapping_non_dict_normalization_path() -> None:
    class _SimpleMapping(Mapping[str, t.GeneralValueType]):
        def __getitem__(self, key: str) -> t.GeneralValueType:
            if key == "a":
                return 1
            raise KeyError(key)

        def __iter__(self) -> Iterator[str]:
            return iter(["a"])

        def __len__(self) -> int:
            return 1

    normalized = u.Generators._normalize_context_to_dict(_SimpleMapping())
    assert normalized == {"a": 1}
