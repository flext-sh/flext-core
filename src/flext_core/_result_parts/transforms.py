"""Monadic transform operations for FlextResult."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Self, cast, overload, override

from flext_core._constants.errors import FlextConstantsErrors as c
from flext_core._models.pydantic import FlextModelsPydantic as mp

from .construction import FlextResultConstructionMixin

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_core._protocols.result import FlextProtocolsResult as p


class FlextResultTransformsMixin[T](FlextResultConstructionMixin[T], ABC):
    """Instance transformation methods for result values and errors."""

    @override
    def filter(self, predicate: Callable[[T], bool]) -> p.Result[T]:
        """Filter success value; returns self or failure if predicate fails."""
        if self.success and self.value is not None:
            try:
                if predicate(self.value):
                    return self
                return self.__class__.fail(c.ERR_RESULT_FILTER_PREDICATE_FAILED)
            except c.EXC_BROAD_RUNTIME as exc:
                return self.__class__.fail(str(exc), exception=exc)
        return self

    @override
    def flat_map[U](self, func: Callable[[T], p.Result[U]]) -> p.Result[U]:
        """Chain operations returning a Result; func produces Result directly."""
        if self.failure:
            # Type bridge: propagated failures adopt the chained payload type.
            result_class = cast("type[FlextResultConstructionMixin[U]]", self.__class__)
            return result_class.fail(
                self.require_error(self),
                error_code=self.error_code,
                error_data=self.error_data,
                exception=self.exception,
            )
        try:
            return self.__class__.from_result(func(self.value))
        except c.EXC_BROAD_RUNTIME as exc:
            # Type bridge: callback exceptions adopt the chained payload type.
            result_class = cast("type[FlextResultConstructionMixin[U]]", self.__class__)
            return result_class.fail(str(exc), exception=exc)

    @override
    def flow_through(self, *funcs: Callable[[T], p.Result[T]]) -> p.Result[T]:
        """Chain multiple homogeneous Result-returning operations in sequence."""
        current = self
        result_class = cast("type[FlextResultConstructionMixin[T]]", self.__class__)
        for func in funcs:
            if current.success:
                result_value = current.value
                if result_value is not None:
                    try:
                        current = self.__class__.from_result(func(result_value))
                    except c.EXC_BROAD_RUNTIME as exc:
                        current = result_class.fail(str(exc), exception=exc)
                else:
                    break
            else:
                break
        return current

    @override
    def fold[U](
        self, on_failure: Callable[[str], U], on_success: Callable[[T], U]
    ) -> U:
        """Catamorphism: reduce result to a single value via callbacks."""
        if self.success and self.value is not None:
            return on_success(self.value)
        return on_failure(self.require_error(self))

    @override
    def lash(self, func: Callable[[str], p.Result[T]]) -> p.Result[T]:
        """Apply recovery function on failure; returns self if success."""
        if self.failure:
            try:
                return self.__class__.from_result(func(self.require_error(self)))
            except c.EXC_BROAD_RUNTIME as exc:
                return self.__class__.fail(str(exc), exception=exc)
        return self

    @override
    def map[U](self, func: Callable[[T], U]) -> p.Result[U]:
        """Transform success value; propagates failure."""
        if self.success:
            try:
                return self.__class__.ok(func(self.value))
            except c.EXC_BROAD_RUNTIME as exc:
                # Type bridge: mapper exceptions adopt the mapped payload type.
                result_class = cast(
                    "type[FlextResultConstructionMixin[U]]", self.__class__
                )
                return result_class.fail(str(exc), exception=exc)
        # Type bridge: propagated failures adopt the mapped payload type.
        result_class = cast("type[FlextResultConstructionMixin[U]]", self.__class__)
        return result_class.fail(
            self.require_error(self),
            error_code=self.error_code,
            error_data=self.error_data,
            exception=self.exception,
        )

    @override
    def map_error(self, func: Callable[[str], str]) -> p.Result[T]:
        """Transform error message; returns self if success."""
        if self.failure:
            try:
                return self.__class__.fail(
                    func(self.require_error(self)),
                    error_code=self.error_code,
                    error_data=self.error_data,
                    exception=self.exception,
                )
            except c.EXC_BROAD_RUNTIME as exc:
                return self.__class__.fail(str(exc), exception=exc)
        return self

    @overload
    def map_or(self, default: None, func: None = None) -> T | None: ...
    @overload
    def map_or[U](self, default: U, func: None = None) -> T | U: ...
    @overload
    def map_or[U](self, default: U, func: Callable[[T], U]) -> U: ...

    @override
    def map_or[U](self, default: U, func: Callable[[T], U] | None = None) -> U | T:
        """Apply func to success value or return default; func optional."""
        if self.success and self.value is not None:
            if func is not None:
                return func(self.value)
            return self.value
        return default

    @override
    def recover[U](self, func: Callable[[str], U]) -> p.Result[T | U]:
        """Recover from failure with fallback value via callback."""
        if self.success:
            value: T | U = self.value
            return self.__class__.ok(value)
        try:
            return self.__class__.ok(func(self.require_error(self)))
        except c.EXC_BROAD_RUNTIME as exc:
            result_class = cast(
                "type[FlextResultConstructionMixin[T | U]]", self.__class__
            )
            return result_class.fail(str(exc), exception=exc)

    @override
    def tap(self, func: Callable[[T], None]) -> p.Result[T]:
        """Apply side effect to success value; return unchanged."""
        if self.success and self.value is not None:
            try:
                func(self.value)
            except c.EXC_BROAD_RUNTIME as exc:
                return self.__class__.fail(str(exc), exception=exc)
        return self

    @override
    def tap_error(self, func: Callable[[str], None]) -> Self:
        """Side effect on failure; return unchanged."""
        if self.failure:
            try:
                func(self.require_error(self))
            except c.EXC_BROAD_RUNTIME as exc:
                return cast("Self", self.__class__.fail(str(exc), exception=exc))
        return self

    @override
    def to_model[U: mp.BaseModel](self, model: type[U]) -> p.Result[U]:
        """Convert success value to Pydantic model; propagates failure."""
        if self.failure:
            # Type bridge: propagated failures adopt the converted model type.
            result_class = cast("type[FlextResultConstructionMixin[U]]", self.__class__)
            return result_class.fail(
                self.require_error(self),
                error_code=self.error_code,
                error_data=self.error_data,
                exception=self.exception,
            )
        try:
            return self.__class__.ok(model.model_validate(self.value))
        except c.EXC_ATTR_RUNTIME_VALIDATION as exc:
            # Type bridge: validation failures adopt the converted model type.
            result_class = cast("type[FlextResultConstructionMixin[U]]", self.__class__)
            return result_class.fail(str(exc), exception=exc)


__all__: list[str] = ["FlextResultTransformsMixin"]
