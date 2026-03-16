from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from pydantic import BaseModel

from flext_core import T, p, r, t
from flext_core._utilities import FlextUtilitiesGuards


class FlextUtilitiesResultHelpers:
    """Result composition helpers (namespace-only, no MRO base).

    Governance: Pure namespace class with only @staticmethod members. No state,
    no instantiation, no fields. BaseModel inheritance not required per §3.1.
    """

    @staticmethod
    def any_(*values: t.NormalizedValue) -> bool:
        return any(bool(v) for v in values)

    @staticmethod
    def empty(items: t.NormalizedValue | None) -> bool:
        if items is None:
            return True
        if not FlextUtilitiesGuards.is_container(items):
            return True
        return not bool(items)

    @staticmethod
    def ends(value: str, suffix: str, *suffixes: str) -> bool:
        return any(value.endswith(s) for s in (suffix, *suffixes))

    @staticmethod
    def err(result: p.Result[T], *, default: str = "Unknown error") -> str:
        if result.is_failure and result.error:
            return str(result.error)
        return default

    @staticmethod
    def not_(value: t.NormalizedValue) -> bool:
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
        func_result = r[T].create_from_callable(func)
        if func_result.is_success:
            return r[T].ok(func_result.value)
        exc = getattr(func_result, "_exception", None)
        if exc is not None and not isinstance(exc, catch):
            raise exc
        if default is not None:
            return r[T].ok(default)
        return r[T].fail(func_result.error or "Callable failed")

    @staticmethod
    def val(result: p.Result[T], *, default: T | None = None) -> r[T]:
        if result.is_success:
            return r[T].ok(result.value)
        if default is not None:
            return r[T].ok(default)
        return r[T].fail(result.error or "Failed to extract result value")

    @staticmethod
    def vals(
        items: Mapping[str, T] | r[Mapping[str, T]],
        *,
        default: list[T] | None = None,
    ) -> r[list[T]]:
        if isinstance(items, r):
            if items.is_failure:
                if default is not None:
                    return r[list[T]].ok(default)
                return r[list[T]].fail(
                    items.error or "Failed to extract values from result"
                )
            value_mapping = items.value
            return r[list[T]].ok(list(value_mapping.values()))
        if items:
            return r[list[T]].ok(list(items.values()))
        if default is not None:
            return r[list[T]].ok(default)
        return r[list[T]].fail("No values available")

    @staticmethod
    def vals_sequence(results: Sequence[p.Result[T]]) -> list[T]:
        return [result.value for result in results if result.is_success]

    @staticmethod
    def ensure_result(
        value: t.NormalizedValue | BaseModel,
    ) -> r[t.NormalizedValue | BaseModel]:
        """Wrap value in r if not already a Result.

        Generic replacement for:
        if not isinstance(val, r): val = r.ok(val)
        """
        return r[t.NormalizedValue | BaseModel].ok(value)


__all__ = ["FlextUtilitiesResultHelpers"]
