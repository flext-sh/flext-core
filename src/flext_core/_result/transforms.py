"""Transform operations for FlextResult.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Self, overload

from pydantic import BaseModel

from flext_core import c, p
from flext_core._result import FlextResultConstruction

if TYPE_CHECKING:
    from collections.abc import Callable


class FlextResultTransforms[T](FlextResultConstruction[T], ABC):
    """Instance transformation methods for result values and errors.

    Covers mapping payloads through callables, mapping errors, combining
    with defaults (``map_or`` family) and converting back to plain values
    or results without changing the success/failure disposition.
    """

    def _as_result(self) -> p.Result[T]:
        """Structural view of self as the abstract Result protocol.

        Returns:
            The resulting ``p.Result[T]``.

        Raises:
            TypeError: If this instance lacks the complete public result family.

        """
        if not isinstance(self, p.Result):
            msg = "Result operations require the complete public result family"
            raise TypeError(msg)
        return self

    def filter(self: Self, predicate: Callable[[T], bool]) -> p.Result[T]:
        """Keep the result only when the predicate holds for the payload.

        A false predicate turns the result into a failure; an exception raised
        by ``predicate`` becomes a failure. Failures pass through unchanged.

        Args:
            predicate: The callable testing the success payload.

        Returns:
            The resulting ``p.Result[T]``.

        """
        if self.success:
            try:
                if predicate(self._payload):
                    return self._as_result()
                return self.__class__.fail(c.ERR_RESULT_FILTER_PREDICATE_FAILED)
            except c.EXC_BROAD_RUNTIME as exc:
                return self.__class__.fail(str(exc), exception=exc)
        return self._as_result()

    def flat_map[U](self: Self, func: Callable[[T], p.Result[U]]) -> p.Result[U]:
        """Chain a result-returning function over the payload.

        Failures propagate with their metadata preserved; an exception raised
        by ``func`` becomes a failure.

        Args:
            func: The callable producing the next result.

        Returns:
            The resulting ``p.Result[U]``.

        """
        if self.failure:
            return self._copy_failure(self._as_result())
        try:
            return self.from_result(func(self._payload))
        except c.EXC_BROAD_RUNTIME as exc:
            return self._failure(str(exc), exception=exc)

    @overload
    def flow_through(self: Self) -> p.Result[T]: ...

    @overload
    def flow_through[U](  # type: ignore
        self: Self,
        first: Callable[[T | U], p.Result[U]],
        /,
        *funcs: Callable[[T | U], p.Result[U]],
    ) -> p.Result[T | U]:
        _ = first, funcs
        raise NotImplementedError

    def flow_through[U](
        self: Self, *funcs: Callable[[T | U], p.Result[U]]
    ) -> p.Result[T | U]:
        """Apply functions left to right, stopping at the first failure.

        Args:
            *funcs: The result-returning functions applied in order.

        Returns:
            The resulting ``p.Result[T | U]``. With no functions, the original
            result is preserved.

        """
        current: p.Result[T | U] = self._as_result()
        for func in funcs:
            if current.success:
                try:
                    current = self.from_result(func(current.value))
                except c.EXC_BROAD_RUNTIME as exc:
                    current = self._failure(str(exc), exception=exc)
            else:
                break
        return current

    def fold[U](
        self: Self,
        on_failure: Callable[[str], U],
        on_success: Callable[[T], U],
    ) -> U:
        """Reduce the result to one value through the matching branch.

        Args:
            on_failure: The callable receiving the error message on failure.
            on_success: The callable receiving the payload on success.

        Returns:
            The value produced by the matching callable.

        """
        if self.success:
            return on_success(self._payload)
        return on_failure(self.require_error(self._as_result()))

    def lash[U](self: Self, func: Callable[[str], p.Result[U]]) -> p.Result[T | U]:
        """Recover a failure through a result-returning error handler.

        Successes pass through unchanged; an exception raised by ``func``
        becomes a failure.

        Args:
            func: The callable mapping the error message to a new result.

        Returns:
            The resulting ``p.Result[T | U]``.

        """
        if self.failure:
            try:
                return self.from_result(func(self.require_error(self._as_result())))
            except c.EXC_BROAD_RUNTIME as exc:
                return self._failure(str(exc), exception=exc)
        return self._as_result()

    def map[U](self: Self, func: Callable[[T], U]) -> p.Result[U]:
        """Transform the payload with a plain function on success.

        Failures propagate with their metadata preserved; an exception raised
        by ``func`` becomes a failure.

        Args:
            func: The callable transforming the success payload.

        Returns:
            The resulting ``p.Result[U]``.

        """
        if self.success:
            try:
                return self._create_success(func(self._payload))
            except c.EXC_BROAD_RUNTIME as exc:
                return self._failure(str(exc), exception=exc)
        return self._copy_failure(self._as_result())

    def map_error(self: Self, func: Callable[[str], str]) -> p.Result[T]:
        """Transform the error message on failure, preserving other metadata.

        Successes pass through unchanged; an exception raised by ``func``
        becomes a failure.

        Args:
            func: The callable transforming the error message.

        Returns:
            The resulting ``p.Result[T]``.

        """
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

    @overload
    def map_or(self, default: None, func: None = None) -> T | None:  # type: ignore
        _ = default, func
        raise NotImplementedError
    @overload
    def map_or[U](self, default: U, func: None = None) -> T | U:  # type: ignore
        _ = default, func
        raise NotImplementedError
    @overload
    def map_or[U](self, default: U, func: Callable[[T], U]) -> U:  # type: ignore
        _ = default, func
        raise NotImplementedError

    def map_or[U](self, default: U, func: Callable[[T], U] | None = None) -> U | T:
        """Apply ``func`` to the payload on success, otherwise return ``default``.

        Args:
            default: The value returned when this result is a failure.
            func: The optional callable applied to the success payload.

        Returns:
            The transformed payload, the payload itself, or ``default``.

        """
        if self.success:
            if func is not None:
                return func(self._payload)
            return self._payload
        return default

    def recover[U](self: Self, func: Callable[[str], U]) -> p.Result[T | U]:
        """Turn a failure into a success through a fallback function.

        Successes pass through unchanged; an exception raised by ``func``
        becomes a failure.

        Args:
            func: The callable mapping the error message to the recovery value.

        Returns:
            The resulting ``p.Result[T | U]``.

        """
        if self.success:
            return self._as_result()
        try:
            return self._create_success(func(self.require_error(self._as_result())))
        except c.EXC_BROAD_RUNTIME as exc:
            return self._failure(str(exc), exception=exc)

    def tap(self: Self, func: Callable[[T], None]) -> p.Result[T]:
        """Run a side effect on the payload without changing the result.

        Only runs on success; an exception raised by ``func`` becomes a
        failure.

        Args:
            func: The callable receiving the success payload.

        Returns:
            The resulting ``p.Result[T]``.

        """
        if self.success:
            try:
                func(self._payload)
            except c.EXC_BROAD_RUNTIME as exc:
                return self.__class__.fail(str(exc), exception=exc)
        return self._as_result()

    def tap_error(self: Self, func: Callable[[str], None]) -> p.Result[T]:
        """Run a side effect on the error message without changing the result.

        Only runs on failure; an exception raised by ``func`` becomes a
        failure.

        Args:
            func: The callable receiving the error message.

        Returns:
            The resulting ``p.Result[T]``.

        """
        if self.failure:
            try:
                func(self.require_error(self._as_result()))
            except c.EXC_BROAD_RUNTIME as exc:
                return self.__class__.fail(str(exc), exception=exc)
        return self._as_result()

    def to_model[U: BaseModel](self: Self, model: type[U]) -> p.Result[U]:
        """Validate the payload into a model instance on success.

        Failures propagate with their metadata preserved; a validation error
        becomes a failure.

        Args:
            model: The model class validating the payload.

        Returns:
            The resulting ``p.Result[U]``.

        """
        if self.failure:
            return self._copy_failure(self._as_result())
        try:
            return self._create_success(model.model_validate(self._payload))
        except c.EXC_ATTR_RUNTIME_VALIDATION as exc:
            return self._failure(str(exc), exception=exc)


__all__: list[str] = ["FlextResultTransforms"]
