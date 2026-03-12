from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar

from flext_core import p, r, t
from flext_core._utilities.guards import FlextUtilitiesGuards

T = TypeVar("T")


class ResultHelpers:
    @staticmethod
    def any_(*values: object) -> bool:
        return any(bool(v) for v in values)

    @staticmethod
    def empty(
        items: Sequence[object] | Mapping[str, object] | str | p.Result[object] | None,
    ) -> bool:
        if FlextUtilitiesGuards.is_result_like(items):
            if items.is_failure:
                return True
            result_value = items.value
            if not FlextUtilitiesGuards.is_container(result_value):
                return True
            return FlextUtilitiesGuards.empty(result_value)
        if items is None:
            return True
        if not FlextUtilitiesGuards.is_container(items):
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
    def not_(value: object) -> bool:
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
    ) -> r[list[T]]:
        if isinstance(items, r):
            if items.is_failure:
                if default is not None:
                    return r[list[T]].ok(default)
                return r[list[T]].fail(
                    items.error or "Failed to extract values from result"
                )
            return r[list[T]].ok(list(cast(Mapping[str, T], items.value).values()))
        if items:
            return r[list[T]].ok(list(items.values()))
        if default is not None:
            return r[list[T]].ok(default)
        return r[list[T]].fail("No values available")

    @staticmethod
    def vals_sequence(results: Sequence[p.Result[T]]) -> list[T]:
        return [result.value for result in results if result.is_success]

    @staticmethod
    def ensure_result[V](value: V | p.Result[V]) -> r[V]:
        """Wrap value in r if not already a Result.

        Generic replacement for:
        if not isinstance(val, r): val = r.ok(val)
        """
        if isinstance(value, r):
            return value

        # Fallback for protocol compliance if it's a Result-like but not FlextResult
        if hasattr(value, "is_success") and hasattr(value, "value"):
            res = cast(p.Result[V], value)
            if res.is_success:
                return r[V].ok(res.value)
            return r[V].fail(res.error)

        return r[V].ok(cast(V, value))


__all__ = ["ResultHelpers"]
