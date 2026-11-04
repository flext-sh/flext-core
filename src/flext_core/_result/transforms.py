"""Monadic transformations, error mapping, and pipeline flow operations for FlextResult.

Provides monadic transformation primitives (``map``, ``flat_map``, ``flow_through``),
recovery combinators (``lash``, ``recover``, ``map_or``), branching operations
(``fold``, ``filter``), side-effect taps (``tap``, ``tap_error``), and Pydantic
model conversions (``to_model``).

See Also:
    - Workspace ADR-001 (Railway-Oriented Programming):
      ``flext/docs/architecture/adr/001-railway-oriented-programming.md``
    - Workspace ADR-027 (Facade Single-Owner Law):
      ``flext/docs/architecture/adr/027-facade-single-owner-law.md``
    - :mod:`flext_core._result.composition`: Sequence and collection combinators.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self, overload

from flext_core import c, p, t
from flext_core._result import FlextResultConstruction

if TYPE_CHECKING:
    from collections.abc import Callable


class FlextResultTransforms[T](FlextResultConstruction[T], ABC):
    """Instance transformation methods for monadic result values and errors.

    Provides functional combinators for projecting payloads, chaining dependent
    railway calls, inspecting values without mutation, and converting errors.

    Type Parameters:
        T: The payload type carried upon successful completion.

    See Also:
        - Workspace ADR-001 (Railway-Oriented Programming):
          ``flext/docs/architecture/adr/001-railway-oriented-programming.md``

    """

    @abstractmethod
    def _as_result(self) -> p.Result[T]:
        """Expose the complete public Result view from the composed MRO owner.

        Returns:
            p.Result[T]: The complete protocol view of this instance.

        """
        raise NotImplementedError

    def filter(self: Self, predicate: Callable[[T], bool]) -> p.Result[T]:
        """Retain the success payload only when the predicate evaluates to True.

        If ``predicate(payload)`` evaluates to False or raises a catchable runtime
        exception, transforms the result into a failure. Failed results pass
        through unchanged.

        Args:
            predicate: Callable taking the payload and returning a boolean.

        Returns:
            p.Result[T]: The original success if predicate holds, otherwise a failure.

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
        """Chain a result-returning callable over the success payload (monadic bind).

        Passes the payload to ``func`` on success, propagating any returned failure
        or newly created failure if ``func`` raises an exception. Failed results pass
        through with original error metadata preserved.

        Args:
            func: Callable producing the subsequent Result given the current payload.

        Returns:
            p.Result[U]: The chained Result instance.

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
    def flow_through[U](
        self: Self,
        first: Callable[[T | U], p.Result[U]],
        /,
        *funcs: Callable[[T | U], p.Result[U]],
    ) -> p.Result[T | U]:
        raise NotImplementedError

    def flow_through[U](
        self: Self, *funcs: Callable[[T | U], p.Result[U]]
    ) -> p.Result[T | U]:
        """Sequentially apply result-returning functions, stopping at the first failure.

        Args:
            *funcs: Variable sequence of result-returning callables applied in order.

        Returns:
            p.Result[T | U]: The cumulative result of the pipeline, or the first
                failure encountered.

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
        """Collapse the result into a single value by executing the matching branch.

        Args:
            on_failure: Callable invoked with the error message if this result failed.
            on_success: Callable invoked with the success payload on success.

        Returns:
            U: The value produced by whichever branch callable was executed.

        """
        if self.success:
            return on_success(self._payload)
        return on_failure(self.require_error(self._as_result()))

    def lash[U](self: Self, func: Callable[[str], p.Result[U]]) -> p.Result[T | U]:
        """Recover from a failure by chaining an error-handling result callable.

        If the current result succeeded, it is returned unchanged. If it failed,
        invokes ``func(error_message)`` and returns its Result.

        Args:
            func: Callable accepting the error message string and returning a Result.

        Returns:
            p.Result[T | U]: The original success result or the recovered Result.

        """
        if self.failure:
            try:
                return self.from_result(func(self.require_error(self._as_result())))
            except c.EXC_BROAD_RUNTIME as exc:
                return self._failure(str(exc), exception=exc)
        return self._as_result()

    def map[U](self: Self, func: Callable[[T], U]) -> p.Result[U]:
        """Transform the success payload by applying a pure mapping function.

        If the current result is a failure, propagates the failure metadata
        without executing ``func``. If ``func`` raises a catchable exception,
        converts it into a failure result.

        Args:
            func: Pure transformation function taking the payload and returning a value.

        Returns:
            p.Result[U]: A new success result with the transformed payload, or failure.

        """
        if self.success:
            try:
                return self._create_success(func(self._payload))
            except c.EXC_BROAD_RUNTIME as exc:
                return self._failure(str(exc), exception=exc)
        return self._copy_failure(self._as_result())

    def map_error(self: Self, func: Callable[[str], str]) -> p.Result[T]:
        """Transform the failure message while preserving other error metadata.

        If the current result succeeded, returns it unchanged. If it failed,
        applies ``func(error_message)`` to update the descriptive error message.

        Args:
            func: Transformation callable mapping the current error string to a new one.

        Returns:
            p.Result[T]: The original success, or an updated failure with new message.

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
    def map_or(self, default: None, func: None = None) -> T | None: ...

    @overload
    def map_or[U](self, default: U, func: None = None) -> T | U: ...

    @overload
    def map_or[U](self, default: U, func: Callable[[T], U]) -> U: ...

    def map_or[U](self, default: U, func: Callable[[T], U] | None = None) -> U | T:
        """Apply ``func`` to payload on success, otherwise return ``default``.

        Args:
            default: Fallback value returned when this result represents a failure.
            func: Optional transformation callable applied to the success payload.

        Returns:
            U | T: The transformed payload, raw payload (if func is None), or default.

        """
        if self.success:
            if func is not None:
                return func(self._payload)
            return self._payload
        return default

    def recover[U](self: Self, func: Callable[[str], U]) -> p.Result[T | U]:
        """Transform a failure into a success by computing a fallback value.

        If the current result succeeded, returns it unchanged. If it failed,
        computes a fallback payload using ``func(error_message)``.

        Args:
            func: Callable mapping the failure error message to a recovery payload.

        Returns:
            p.Result[T | U]: The original success result or a newly wrapped success.

        """
        if self.success:
            return self._as_result()
        try:
            return self._create_success(func(self.require_error(self._as_result())))
        except c.EXC_BROAD_RUNTIME as exc:
            return self._failure(str(exc), exception=exc)

    def tap(self: Self, func: Callable[[T], None]) -> p.Result[T]:
        """Execute a side effect on the success payload without altering the result.

        Args:
            func: Callable accepting the success payload for inspection or logging.

        Returns:
            p.Result[T]: The original result instance, or a failure if ``func`` raises.

        """
        if self.success:
            try:
                func(self._payload)
            except c.EXC_BROAD_RUNTIME as exc:
                return self.__class__.fail(str(exc), exception=exc)
        return self._as_result()

    def tap_error(self: Self, func: Callable[[str], None]) -> p.Result[T]:
        """Execute a side effect on the failure error message without altering result.

        Args:
            func: Callable accepting the error message string for inspection or logging.

        Returns:
            p.Result[T]: The original result instance, or a failure if ``func`` raises.

        """
        if self.failure:
            try:
                func(self.require_error(self._as_result()))
            except c.EXC_BROAD_RUNTIME as exc:
                return self.__class__.fail(str(exc), exception=exc)
        return self._as_result()

    def to_model[U: t.BaseModelType](self: Self, model: type[U]) -> p.Result[U]:
        """Validate and parse the success payload into an instance of a Pydantic model.

        Args:
            model: Target Pydantic model class performing payload validation.

        Returns:
            p.Result[U]: A success result carrying the validated model instance,
                or a failure result capturing validation errors.

        """
        if self.failure:
            return self._copy_failure(self._as_result())
        try:
            return self._create_success(model.model_validate(self._payload))
        except c.EXC_ATTR_RUNTIME_VALIDATION as exc:
            return self._failure(str(exc), exception=exc)


__all__: list[str] = ["FlextResultTransforms"]
