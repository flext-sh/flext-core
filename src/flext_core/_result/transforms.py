"""Transform operations for FlextResult."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel

from flext_core import c

from .construction import FlextResultConstruction

if TYPE_CHECKING:
    from flext_core import p


class FlextResultTransforms[T](FlextResultConstruction[T]):
    """Instance transformation methods for result values and errors."""

    def _as_result(self) -> p.Result[T]:
        """Structural cast of self to p.Result for internal calls."""
        return cast("p.Result[T]", self)

    def filter(self, predicate: Callable[[T], bool]) -> p.Result[T]:
        if self.success:
            try:
                if predicate(self._payload):
                    return self._as_result()
                return self.__class__.fail(c.ERR_RESULT_FILTER_PREDICATE_FAILED)
            except c.EXC_BROAD_RUNTIME as exc:
                return self.__class__.fail(str(exc), exception=exc)
        return self._as_result()

    def flat_map[U](self, func: Callable[[T], p.Result[U]]) -> p.Result[U]:
        if self.failure:
            return self.__class__.fail(
                self.require_error(self._as_result()),
                error_code=self.error_code,
                error_data=self.error_data,
                exception=self._exception,
            )
        try:
            return self.__class__.from_result(func(self._payload))
        except c.EXC_BROAD_RUNTIME as exc:
            return self.__class__.fail(str(exc), exception=exc)

    def flow_through[U](self, *funcs: Callable[..., p.Result[U]]) -> p.Result[T | U]:
        current: p.Result[T | U] = self._as_result()
        for func in funcs:
            if current.success:
                try:
                    current = current.__class__.from_result(func(current.value))
                except c.EXC_BROAD_RUNTIME as exc:
                    current = current.__class__.fail(str(exc), exception=exc)
            else:
                break
        return current

    def fold[U](
        self, on_failure: Callable[[str], U], on_success: Callable[[T], U]
    ) -> U:
        if self.success:
            return on_success(self._payload)
        return on_failure(self.require_error(self._as_result()))

    def lash[U](self, func: Callable[[str], p.Result[U]]) -> p.Result[T | U]:
        if self.failure:
            try:
                return self.__class__.from_result(
                    func(self.require_error(self._as_result()))
                )
            except c.EXC_BROAD_RUNTIME as exc:
                return self.__class__.fail(str(exc), exception=exc)
        return self._as_result()

    def map[U](self, func: Callable[[T], U]) -> p.Result[U]:
        if self.success:
            try:
                return self.__class__.ok(func(self._payload))
            except c.EXC_BROAD_RUNTIME as exc:
                return self.__class__.fail(str(exc), exception=exc)
        return self.__class__.fail(
            self.require_error(self._as_result()),
            error_code=self.error_code,
            error_data=self.error_data,
            exception=self._exception,
        )

    def map_error(self, func: Callable[[str], str]) -> p.Result[T]:
        if self.failure:
            try:
                return self.__class__.fail(
                    func(self.require_error(self._as_result())),
                    error_code=self.error_code,
                    error_data=self.error_data,
                    exception=self._exception,
                )
            except c.EXC_BROAD_RUNTIME as exc:
                return self.__class__.fail(str(exc), exception=exc)
        return self._as_result()

    def map_or[U](self, default: U, func: Callable[..., U] | None = None) -> U | T:
        if self.success:
            if func is not None:
                return func(self._payload)
            return self._payload
        return default

    def recover[U](self, func: Callable[[str], U]) -> p.Result[T | U]:
        if self.success:
            return self._as_result()
        try:
            return self.__class__.ok(func(self.require_error(self._as_result())))
        except c.EXC_BROAD_RUNTIME as exc:
            return self.__class__.fail(str(exc), exception=exc)

    def tap(self, func: Callable[[T], None]) -> p.Result[T]:
        if self.success:
            try:
                func(self._payload)
            except c.EXC_BROAD_RUNTIME as exc:
                return self.__class__.fail(str(exc), exception=exc)
        return self._as_result()

    def tap_error(self, func: Callable[[str], None]) -> p.Result[T]:
        if self.failure:
            try:
                func(self.require_error(self._as_result()))
            except c.EXC_BROAD_RUNTIME as exc:
                return self.__class__.fail(str(exc), exception=exc)
        return self._as_result()

    def to_model[U: BaseModel](self, model: type[U]) -> p.Result[U]:
        if self.failure:
            return self.__class__.fail(
                self.require_error(self._as_result()),
                error_code=self.error_code,
                error_data=self.error_data,
                exception=self._exception,
            )
        try:
            return self.__class__.ok(model.model_validate(self._payload))
        except c.EXC_ATTR_RUNTIME_VALIDATION as exc:
            return self.__class__.fail(str(exc), exception=exc)


__all__: list[str] = ["FlextResultTransforms"]
