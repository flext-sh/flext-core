from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar

from flext_core._utilities.collection import FlextUtilitiesCollection
from flext_core._utilities.guards import FlextUtilitiesGuards
from flext_core.protocols import p
from flext_core.result import r
from flext_core.typings import t

T = TypeVar("T")
R2 = TypeVar("R2")


class ResultHelpers:
    @staticmethod
    def err(result: p.Result[T], *, default: str = "Unknown error") -> str:
        if result.is_failure and result.error:
            return str(result.error)
        return default

    @staticmethod
    def val(result: p.Result[T], *, default: T | None = None) -> T | None:
        if result.is_success:
            return result.value
        return default

    @staticmethod
    def vals(
        items: Mapping[str, T] | r[Mapping[str, T]],
        *,
        default: list[T] | None = None,
    ) -> list[T]:
        if isinstance(items, r):
            if items.is_failure:
                return default if default is not None else []
            return list(items.value.values())
        return (
            list(items.values()) if items else (default if default is not None else [])
        )

    @staticmethod
    def vals_sequence(results: Sequence[p.Result[T]]) -> list[T]:
        return [result.value for result in results if result.is_success]

    @staticmethod
    def or_(*values: T | None, default: T | None = None) -> T | None:
        for value in values:
            if value is not None:
                return value
        return default

    @staticmethod
    def try_(
        func: Callable[[], T],
        *,
        default: T | None = None,
        catch: type[Exception] | tuple[type[Exception], ...] = Exception,
    ) -> T | None:
        try:
            return func()
        except Exception as exc:
            if isinstance(exc, catch):
                return default
            raise

    @staticmethod
    def starts(value: str, prefix: str, *prefixes: str) -> bool:
        return any(value.startswith(p) for p in (prefix, *prefixes))

    @staticmethod
    def not_(value: t.ConfigMapValue) -> bool:
        return not bool(value)

    @staticmethod
    def any_(*values: t.ConfigMapValue) -> bool:
        return any(bool(v) for v in values)

    @staticmethod
    def empty(items: object) -> bool:
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
    def count(
        items: Sequence[t.ConfigMapValue] | Mapping[str, t.ConfigMapValue],
        predicate: Callable[[t.ConfigMapValue], bool] | None = None,
    ) -> int:
        if predicate is None:
            return len(items)
        if isinstance(items, Mapping):
            return FlextUtilitiesCollection.count(list(items.values()), predicate)
        return FlextUtilitiesCollection.count(items, predicate)

    @staticmethod
    def ends(value: str, suffix: str, *suffixes: str) -> bool:
        return any(value.endswith(s) for s in (suffix, *suffixes))


__all__ = ["ResultHelpers"]
