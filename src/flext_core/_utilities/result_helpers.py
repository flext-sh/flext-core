from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar

from flext_core import p, r, t
from flext_core._utilities.collection import FlextUtilitiesCollection
from flext_core._utilities.guards import FlextUtilitiesGuards

T = TypeVar("T")


class ResultHelpers:
    @staticmethod
    def any_(*values: t.ContainerValue) -> bool:
        return any(bool(v) for v in values)

    @staticmethod
    def count(
        items: Sequence[t.ContainerValue] | Mapping[str, t.ContainerValue],
        predicate: Callable[[t.ContainerValue], bool] | None = None,
    ) -> int:
        if predicate is None:
            return len(items)
        if isinstance(items, Mapping):
            return FlextUtilitiesCollection.count(list(items.values()), predicate)
        return FlextUtilitiesCollection.count(items, predicate)

    @staticmethod
    def empty(
        items: Sequence[t.ContainerValue] | Mapping[str, t.ContainerValue] | str | None,
    ) -> bool:
        if isinstance(items, r):
            if items.is_failure:
                return True
            return FlextUtilitiesGuards.empty(items.value)
        if items is None:
            return True
        if not FlextUtilitiesGuards.is_general_value_type(items):
            return True
        return FlextUtilitiesGuards.empty(items)

    @staticmethod
    def ends(value: str, suffix: str, *suffixes: str) -> bool:
        return any(value.endswith(s) for s in (suffix, *suffixes))

    @staticmethod
    def err(result: p.Result[T], *, default: str = "Unknown error") -> str:
        if result.is_failure and result.error:
            return str(result.error)
        return default

    @staticmethod
    def not_(value: t.ContainerValue) -> bool:
        return not bool(value)

    @staticmethod
    def or_(*values: T | None, default: T | None = None) -> r[T]:
        for value in values:
            if value is not None:
                return r[T].ok(value)
        if default is not None:
            return r[T].ok(default)
        return r[T].fail("No non-None value found")

    @staticmethod
    def starts(value: str, prefix: str, *prefixes: str) -> bool:
        return any(value.startswith(p) for p in (prefix, *prefixes))

    @staticmethod
    def try_(
        func: Callable[[], T],
        *,
        default: T | None = None,
        catch: type[Exception] | tuple[type[Exception], ...] = Exception,
    ) -> r[T]:
        try:
            return r[T].ok(func())
        except Exception as exc:
            if isinstance(exc, catch):
                if default is not None:
                    return r[T].ok(default)
                return r[T].fail(str(exc))
            raise

    @staticmethod
    def val(result: p.Result[T], *, default: T | None = None) -> r[T]:
        if result.is_success:
            return r[T].ok(result.value)
        if default is not None:
            return r[T].ok(default)
        return r[T].fail(result.error or "Failed to extract result value")

    @staticmethod
    def vals(
        items: Mapping[str, T] | r[Mapping[str, T]], *, default: list[T] | None = None
    ) -> list[T]:
        if isinstance(items, r):
            if items.is_failure:
                return default if default is not None else []
            return list(items.value.values())
        return list(items.values()) if items else default if default is not None else []

    @staticmethod
    def vals_sequence(results: Sequence[p.Result[T]]) -> list[T]:
        return [result.value for result in results if result.is_success]


__all__ = ["ResultHelpers"]
