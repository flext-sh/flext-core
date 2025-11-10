"""Railway-oriented result type for type-safe error handling.

This module provides FlextResult[T], a type-safe result type implementing
the railway pattern (Either monad) for explicit error handling throughout
the FLEXT ecosystem.

FlextResult wraps operation outcomes with explicit success/failure states,
providing monadic operations for functional composition and safe error
propagation without exceptions.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import types
from collections.abc import Callable, Iterator, Sequence
from contextlib import suppress
from typing import Never, Self, cast, overload, override

from beartype.door import is_bearable
from returns.io import IO, IOFailure, IOResult, IOSuccess
from returns.maybe import Maybe, Nothing, Some
from returns.result import Failure, Result, Success, safe

from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions


class FlextResult[T_co]:
    """FlextResult[T_co] type-safe result type implementing the railway pattern (Either monad).

    Core Foundation Pattern: Railway-Oriented Programming
    ======================================================
    Provides monadic operations for functional error handling with explicit
    success/failure states. Wraps operation outcomes and enables composable
    error propagation without exceptions in business logic.

    Structural Typing and Protocol Compliance:
    ===========================================
    FlextResult satisfies FlextProtocols.Result interface through structural
    typing (duck typing) by implementing required methods:
    - ok(data) - Create successful result wrapping data
    - fail(error) - Create failed result with error message
    - is_success / is_failure - Check result state
    - map(func) - Transform success value
    - flat_map(func) - Chain operations returning FlextResult
    - unwrap() / value - Extract success value
    - error / error_code - Access error information
    - pipeline() - Compose multiple operations

    Key Features:
    =============
    - Type-safe success/failure wrapping with generic [T_co] covariance
    - Monadic operations: map, flat_map, filter, bind, traverse, lash
    - Railway pattern for functional composition
    - Context managers for resource management
    - Advanced combinators: alt, accumulate_errors, parallel_map, with_resource
    - Integration with returns library for battle-tested operations
    - Error metadata: error_code, error_data for structured logging
    - Utility methods: collect_successes, batch_process, safe_call
    - Maybe/IO interoperability for functional ecosystem integration

    Architecture Integration:
    ========================
    - Layer 1 (Foundation): Core result monad implementation
    - Used by: ALL layers (2-4) for operation composition
    - Dependencies: FlextConstants (error codes), FlextExceptions
    - Integration: FlextLogger (structured error logging)
    - Ecosystem: 32+ projects depend on FlextResult[T] for error handling

    Monadic Operations (Railway Pattern):
    =====================================
    FlextResult implements the following monadic pattern:
    1. ok(data: T) -> FlextResult[T]: Create successful result
    2. fail(error: str, ...) -> FlextResult[T]: Create failed result
    3. map(func: T -> U) -> FlextResult[U]: Transform success
    4. flat_map(func: T -> FlextResult[U]) -> FlextResult[U]: Chain operations
    5. filter(predicate: T -> bool) -> FlextResult[T]: Conditional filter

    Railway composition prevents exceptions through explicit error propagation:
    - Each operation takes the railway: either success or failure track
    - Failures short-circuit remaining operations
    - No exception catching needed in application code

    Usage Pattern:
    ==============
        >>> from flext_core import FlextResult
        >>>
        >>> # Basic success case
        >>> result = FlextResult[int].ok(42)
        >>> if result.is_success:
        ...     value = result.unwrap()
        >>>
        >>> # Basic failure case
        >>> result = FlextResult[int].fail("Operation failed", error_code="API_ERROR")
        >>> if result.is_failure:
        ...     error = result.error
        >>>
        >>> # Railway composition - chaining operations
        >>> def validate(x: int) -> FlextResult[int]:
        ...     return (
        ...         FlextResult[int].ok(x)
        ...         if x > 0
        ...         else FlextResult[int].fail("Not positive")
        ...     )
        >>> def double(x: int) -> FlextResult[int]:
        ...     return FlextResult[int].ok(x * 2)
        >>>
        >>> # Compose operations - short-circuits on first failure
        >>> final = (
        ...     FlextResult[int]
        ...     .ok(5)
        ...     .flat_map(validate)
        ...     .flat_map(double)
        ...     .map(lambda x: x + 1)
        ... )
        >>> # final is Success(21) = ((5 * 2) + 1)

    Advanced Patterns:
    ==================
    - flow_through(*funcs): Compose multiple operations sequentially
    - lash(func): Error recovery function (opposite of flat_map)
    - alt(default): Alternative result on failure
    - traverse(items, func): Map over list with failure propagation
    - pipeline(initial, *ops): Compose operations with initial value
    - accumulate_errors(*results): Collect multiple errors
    - parallel_map(items, func): Map with fail-fast or collect-all
    - with_resource(factory, op, cleanup): Resource management

    API Access Methods:
    ===================
    Multiple ways to access result data:
    - result.value - Get success value (raises on failure)
    - result.unwrap() - Get success value or raise error
    - result.value_or_none - Get value or None
    - result[0] - Tuple unpacking: value or None
    - result[1] - Tuple unpacking: error or None
    - result | default - Operator overload for default values

    Error Information Structure:
    ===========================
    - error: Human-readable error message (required)
    - error_code: Categorized error code from FlextConstants.Errors
    - error_data: Additional metadata dictionary for observability
    - Example: fail("API timeout", error_code="TIMEOUT_ERROR", error_data={"timeout_ms": 5000})

    Interoperability:
    =================
    - to_maybe() / from_maybe(): Convert to/from returns.maybe.Maybe
    - to_io() / to_io_result(): Convert to returns.io types
    - from_callable(): Wrap functions that might raise exceptions
    - safe_call(): Execute callable with automatic exception handling

    Advanced Type Features:
    =======================
    - Generic covariance [T_co]: Enables proper type hierarchy
    - Python 3.13 discriminated union: Pattern matching support
    - Type operators: >> (flat_map), << (map), % (filter), ^ (recover)
    - Pattern matching: match result: case FlextResult(_data=v, _error=None)

    Integration Examples:
    ======================
    # Service operation with FlextResult
    from flext_core import FlextResult, FlextService

    class UserService(FlextService):
        def create_user(self, data: dict) -> FlextResult[User]:
            if not data.get("email"):
                return FlextResult[User].fail("Email required")
            user = User(**data)
            return FlextResult[User].ok(user)

    # Handler with FlextResult
    class CreateUserHandler:
        def execute(self, command: CreateUserCommand) -> FlextResult[User]:
            return self.service.create_user(command.data)

    # Pipeline composition
    def process_users(raw_users: list[dict]) -> FlextResult[list[User]]:
        return FlextResult.traverse(
            raw_users,
            lambda d: user_service.create_user(d)
        )
    """

    # Internal storage using returns.Result as backend
    _result: Result[T_co, str]
    _error_code: str | None
    _error_data: dict[str, object]

    # Runtime type validation attribute (set by __class_getitem__)
    _expected_type: type | None = None

    # =========================================================================
    # PRIVATE MEMBERS - Internal helpers to avoid circular imports
    # =========================================================================

    @staticmethod
    def _get_exceptions() -> type[FlextExceptions]:
        """Get FlextExceptions class."""
        return FlextExceptions

    # Overloaded constructor for proper type discrimination.
    @overload
    def __init__(
        self,
        *,
        data: T_co,
        error: None = None,
        error_code: None = None,
        error_data: None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        data: None = None,
        error: str,
        error_code: str | None = None,
        error_data: dict[str, object] | None = None,
    ) -> None: ...

    def __init__(  # noqa: C901  # Complex type validation with multiple fallbacks required
        self,
        *,
        data: T_co | None = None,
        error: str | None = None,
        error_code: str | None = None,
        error_data: dict[str, object] | None = None,
    ) -> None:
        """Initialize result with either success data or error using returns.Result backend."""
        super().__init__()

        # Runtime type validation if _expected_type is set (via __class_getitem__)
        if self._expected_type is not None and error is None and data is not None:
            # Primary validation: use beartype for type checking
            is_valid = is_bearable(data, self._expected_type)

            # Fallback: if beartype fails and expected_type is a class,
            # check if data is an instance (handles generic class subclasses)
            if not is_valid and isinstance(self._expected_type, type):
                # isinstance can fail with some generic types
                with suppress(TypeError):
                    is_valid = isinstance(data, self._expected_type)

                # Second fallback: check if data is instance of base class
                # This handles cases like FlextHandlers[object, object] where
                # data is ConcreteHandler(FlextHandlers) - subclass of base class
                if not is_valid:
                    # Try to find base classes in __orig_bases__ or __mro__
                    orig_bases = getattr(self._expected_type, "__orig_bases__", ())
                    for base in orig_bases:
                        # Get the origin (e.g., FlextHandlers from FlextHandlers[T, R])
                        origin = getattr(base, "__origin__", base)
                        if origin is None or not isinstance(origin, type):
                            continue
                        with suppress(TypeError):
                            if isinstance(data, origin):
                                is_valid = True
                                break

                    # If still not valid, try __mro__ (method resolution order)
                    if not is_valid:
                        mro = getattr(self._expected_type, "__mro__", ())
                        for base_class in mro:
                            if base_class is object or not isinstance(base_class, type):
                                continue
                            with suppress(TypeError):
                                if isinstance(data, base_class):
                                    is_valid = True
                                    break

            if not is_valid:
                expected_name = getattr(
                    self._expected_type, "__name__", str(self._expected_type)
                )
                actual_name = type(data).__name__
                msg = (
                    f"FlextResult[{expected_name}].ok() received {actual_name} "
                    f"instead of {expected_name}. Data: {data!r}"
                )
                raise TypeError(msg)

        # Architectural invariant: exactly one of data or error must be provided.
        if error is not None:
            # Failure path: create Failure with error message
            self._result = Failure(error)
        else:
            # Success path: create Success with data
            # Note: None is a valid success value (e.g., FlextResult[None].ok(None))
            # The returns library supports Success(None) for void/unit operations
            self._result = Success(cast("T_co", data))

        self._error_code = error_code
        self._error_data = error_data or {}
        self.metadata: object | None = None

    def __class_getitem__(cls, item: type) -> type[Self]:
        """Intercept FlextResult[T] to create typed subclass for runtime validation.

        When FlextResult[int] is accessed, this method creates a subclass that
        stores the expected type (int) in _expected_type. The subclass inherits
        all methods and behaviors but adds automatic type validation in __init__.

        This enables automatic type checking at runtime:
            FlextResult[int].ok(42)       # ✅ Valid
            FlextResult[int].ok("string") # ❌ TypeError

        Args:
            item: The type parameter (e.g., int, str, User)

        Returns:
            A typed subclass with _expected_type set

        Example:
            >>> result = FlextResult[int].ok(42)  # ✅ Passes
            >>> result = FlextResult[int].ok("string")  # ❌ TypeError

        """
        if item is cls or (hasattr(item, "__origin__") and item.__origin__ is cls):
            return cls
        try:
            if isinstance(item, type) and issubclass(item, cls):
                return item
        except TypeError:
            pass

        # Create typed subclass dynamically with proper type annotations
        # This is a valid pattern for Generic class subscription
        cls_name = getattr(cls, "__name__", "FlextResult")
        cls_qualname = getattr(cls, "__qualname__", "FlextResult")
        type_name = getattr(item, "__name__", str(item))

        # Create subclass using dynamic type() for Generic subscription
        # NOTE: type: ignore required here due to type checker limitation with metaprogramming
        # This is valid Python metaprogramming - dynamically creating a class at runtime
        # Type checkers cannot verify dynamic type() calls with 3 arguments
        typed_subclass = type(
            f"{cls_name}[{type_name}]",
            (cls,),
            {"_expected_type": item},
        )

        # Preserve qualname for better debugging
        typed_subclass.__qualname__ = f"{cls_qualname}[{type_name}]"

        return typed_subclass

    @property
    def is_success(self) -> bool:
        """Return ``True`` when the result carries a successful payload."""
        return isinstance(self._result, Success)

    @property
    def success(self) -> bool:
        """Return ``True`` when the result carries a successful payload, alternative name for is_success."""
        return isinstance(self._result, Success)

    @property
    def is_failure(self) -> bool:
        """Return ``True`` when the result represents a failure."""
        return isinstance(self._result, Failure)

    @property
    def failed(self) -> bool:
        """Return ``True`` when the result represents a failure, alternative name for is_failure."""
        return isinstance(self._result, Failure)

    @property
    def value(self) -> T_co:
        """Return the success payload, raising :class:`ValidationError` on failure."""
        if self.is_failure:
            msg = "Attempted to access value on failed result"
            raise FlextResult._get_exceptions().ValidationError(
                message=msg, error_code="VALIDATION_ERROR"
            )
        # Use the returns backend to unwrap the value
        return self._result.unwrap()

    @property
    def error(self) -> str | None:
        """Return the captured error message for failure results."""
        if self.is_success:
            return None
        # Extract error from returns.Result backend using failure()
        return self._result.failure()

    @property
    def error_code(self) -> str | None:
        """Return the structured error code supplied on failure."""
        return self._error_code

    @property
    def error_data(self) -> dict[str, object]:
        """Return the structured error metadata dictionary for observability."""
        return self._error_data

    @property
    def _data(self) -> T_co | None:
        """Internal property to access result data from returns.Result backend."""
        return self._result.unwrap() if self.is_success else None

    @property
    def _error(self) -> str | None:
        """Internal property to access error message from returns.Result backend."""
        return self._result.failure() if self.is_failure else None

    @classmethod
    def ok(cls, data: T_co) -> Self:
        """Create a successful FlextResult wrapping the provided data.

        This is the primary way to create successful results throughout the
        FLEXT ecosystem. Use FlextResult.ok() for all successful operations.

        **Runtime Type Checking**: Enable via FlextRuntime.enable_runtime_checking()

        Args:
            data: The successful data to wrap in the result.

        Returns:
            Self: A successful FlextResult containing the provided data.

        Example:
            ```python
            from flext_core import FlextResult
            from pydantic import EmailStr


            def validate_user_email(email: EmailStr) -> FlextResult[str]:
                # Pydantic v2 EmailStr validates format natively
                return FlextResult[str].ok(email)  # Success case
            ```

        """
        return cls(data=data)

    # Note: Classmethod `success()` removed to avoid name collision with
    # the instance property `success`. Use `ok()` instead.

    @classmethod
    def fail(
        cls,
        error: str | None,
        /,
        *,
        error_code: str | None = None,
        error_data: dict[str, object] | None = None,
    ) -> Self:
        """Create a failed FlextResult with structured error information.

        This is the primary way to create failed results throughout the FLEXT
        ecosystem. Use FlextResult.fail() for all error conditions instead of
        raising exceptions in business logic.

        Args:
            error: The error message describing the failure.
            error_code: Optional error code for categorization and monitoring.
            error_data: Optional additional error data/metadata for diagnostics.

        Returns:
            Self: A failed FlextResult with the provided error information.

        Example:
            ```python
            from flext_core import FlextResult


            def divide_numbers(a: int, b: int) -> FlextResult[float]:
                if b == 0:
                    return FlextResult[float].fail(
                        "Division by zero not allowed",
                        error_code="MATH_ERROR",
                        error_data={"dividend": a, "divisor": b},
                    )
                return FlextResult[float].ok(a / b)
            ```

        """
        # Normalize empty/whitespace errors to default message
        if not error or error.isspace():
            actual_error = "Unknown error occurred"
        else:
            actual_error = error

        # Create a new instance with the correct type annotation
        return cls(
            error=actual_error,
            error_code=error_code,
            error_data=error_data,
        )

    @classmethod
    def from_callable(
        cls: type[FlextResult[T_co]],
        func: Callable[[], T_co],
        *,
        error_code: str | None = None,
    ) -> FlextResult[T_co]:
        """Create a FlextResult from a callable using returns library @safe decorator.

        This method automatically wraps exceptions from the callable into a FlextResult
        failure, using the battle-tested @safe decorator from the returns library.
        This replaces manual try/except patterns with functional composition.

        Args:
            func: Callable that returns T_co (may raise exceptions)
            error_code: Optional error code for failures (defaults to OPERATION_ERROR)

        Returns:
            FlextResult[T_co] wrapping the function result or any exception

        Example:
            ```python
            from flext_core import FlextResult


            def risky_operation() -> dict[str, object]:
                return api.fetch_data()  # May raise exceptions


            # Old pattern (manual try/except)
            def old_way() -> FlextResult[dict]:
                try:
                    data = risky_operation()
                    return FlextResult[dict].ok(data)
                except Exception as e:
                    return FlextResult[dict].fail(str(e))


            # New pattern (using from_callable with @safe)
            result = FlextResult[dict].from_callable(risky_operation)

            # With custom error code
            result = FlextResult[dict].from_callable(
                risky_operation, error_code="API_ERROR"
            )
            ```

        """
        # Use @safe to wrap the callable - converts exceptions to Result
        safe_func = safe(func)
        returns_result = safe_func()

        # Check if it's a Success or Failure using isinstance
        if isinstance(returns_result, Success):
            # Success case - extract value using unwrap()
            value = returns_result.unwrap()
            return cls.ok(value)
        if isinstance(returns_result, Failure):
            # Failure case - extract exception using failure()
            exception = returns_result.failure()
            error_msg = str(exception) if exception else "Callable execution failed"
            return cls.fail(
                error_msg,
                error_code=error_code or FlextConstants.Errors.OPERATION_ERROR,
            )
        # Should never reach here, but handle just in case
        return cls.fail(
            "Unexpected result type from callable",
            error_code=error_code or FlextConstants.Errors.OPERATION_ERROR,
        )

    def flow_through(
        self, *functions: Callable[[T_co], FlextResult[T_co]]
    ) -> FlextResult[T_co]:
        """Compose multiple operations into a flow using returns library patterns.

        This method enables functional composition of operations on a FlextResult,
        short-circuiting on the first failure. It uses the flow pattern from the
        returns library for proper railway-oriented programming.

        Args:
            *functions: Variable number of functions that take T_co and return FlextResult[T_co]

        Returns:
            FlextResult[T_co] after all operations or first failure

        Example:
            ```python
            from flext_core import FlextResult


            def validate(data: dict) -> FlextResult[dict]:
                if not data:
                    return FlextResult[dict].fail("Empty data")
                return FlextResult[dict].ok(data)


            def enrich(data: dict) -> FlextResult[dict]:
                data["enriched"] = True
                return FlextResult[dict].ok(data)


            def save(data: dict) -> FlextResult[dict]:
                # Save to database
                return FlextResult[dict].ok(data)


            # Traditional chaining (verbose)
            result = validate(data).flat_map(enrich).flat_map(save)

            # Flow pattern (cleaner - inspired by returns.pipeline.flow)
            result = (
                FlextResult[dict]
                .ok(data)
                .flow_through(
                    validate,
                    enrich,
                    save,
                )
            )
            ```

        """
        if self.is_failure:
            return self

        current_result: FlextResult[T_co] = self
        for func in functions:
            current_result = current_result.flat_map(func)
            if current_result.is_failure:
                return current_result

        return current_result

    # Operations
    def map[U](self, func: Callable[[T_co], U]) -> FlextResult[U]:
        """Transform the success payload using ``func`` while preserving errors.

        Delegates to returns.Result.map() for monadic operations.

        Args:
            func: Function to transform the success value

        Returns:
            New FlextResult with transformed value or original error

        """
        try:
            # Use returns.Result.map() for type-safe transformation
            mapped_result = self._result.map(func)

            # Convert back to FlextResult while preserving error metadata
            if isinstance(mapped_result, Success):
                return FlextResult[U].ok(mapped_result.unwrap())
            return FlextResult[U].fail(
                mapped_result.failure(),
                error_code=self._error_code,
                error_data=self._error_data,
            )

        except Exception as e:
            # VALIDATION HIERARCHY - User callable exception handling (CRITICAL)
            # Level 1: Specific known exceptions (TypeError, AttributeError)
            # Level 2: Common application exceptions (ValueError, RuntimeError, etc.)
            # Level 3: Catch-all for custom exceptions from user functions
            # Framework code executing user callables MUST catch all exception types
            return FlextResult[U].fail(
                f"Transformation failed: {e}",
                error_code=FlextConstants.Errors.MAP_ERROR,
                error_data={
                    "exception_type": type(e).__name__,
                    "exception": str(e),
                },
            )

    def flat_map[U](self, func: Callable[[T_co], FlextResult[U]]) -> FlextResult[U]:
        """Chain operations returning FlextResult.

        Delegates to returns.Result.bind() for monadic bind operation.
        """
        try:
            # Extract value from our internal backend and apply func
            if isinstance(self._result, Failure):
                return FlextResult[U].fail(
                    self._result.failure(),
                    error_code=self._error_code,
                    error_data=self._error_data,
                )

            value = self._result.unwrap()
            # Apply the function which returns FlextResult
            result_u: FlextResult[U] = func(value)

            # Return the FlextResult directly (already in our format)
            return result_u

        except Exception as e:
            # VALIDATION HIERARCHY - User callable exception handling
            # Level 1: Specific known exceptions (TypeError, ValueError, KeyError)
            # Level 2: Common application exceptions (RuntimeError, IndexError)
            # Level 3: Catch-all for custom exceptions from user functions
            # Framework code executing user callables MUST catch all exception types
            return FlextResult[U].fail(
                f"Flat map operation failed: {e}",
                error_code=FlextConstants.Errors.BIND_ERROR,
                error_data={
                    "exception_type": type(e).__name__,
                    "exception": str(e),
                },
            )

    def bind[U](self, func: Callable[[T_co], FlextResult[U]]) -> FlextResult[U]:
        """Monadic bind operation (alias for flat_map).

        Part of Monad[T] protocol implementation.
        Delegates to flat_map() for actual implementation.

        Args:
            func: Function returning FlextResult[U]

        Returns:
            FlextResult[U]: Result of applying function to wrapped value

        """
        return self.flat_map(func)

    def __bool__(self) -> bool:
        """Return True if successful, False if failed."""
        return self.is_success

    def __iter__(self) -> Iterator[T_co | str | None]:
        """Enable unpacking: value, error = result."""
        if self.is_success:
            yield self._result.unwrap()
            yield None
        else:
            yield None
            yield self._result.failure()

    def __getitem__(self, key: int) -> T_co | str | None:
        """Access result[0] for data, result[1] for error."""
        if key == 0:
            return self._result.unwrap() if self.is_success else None
        if key == 1:
            return self._result.failure() if self.is_failure else None
        msg = "FlextResult only supports indices 0 (data) and 1 (error)"
        raise FlextResult._get_exceptions().NotFoundError(
            msg, resource_type=f"index[{key}]"
        )

    def __or__(self, default: T_co) -> T_co:
        """Use | operator for default values: result | default_value.."""
        if self.is_success:
            if self._data is None:
                return default  # Handle None data case
            return self._data
        return default

    def __enter__(self) -> T_co:
        """Context manager entry - returns value or raises on error."""
        if self.is_failure:
            error_msg = self._error or "Context manager failed"
            raise FlextResult._get_exceptions().BaseError(
                message=error_msg, error_code="OPERATION_ERROR"
            )

        return cast("T_co", self._data)

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: types.TracebackType | None,
    ) -> None:
        """Context manager exit."""
        # Parameters available for future error handling logic
        return

    @property
    def data(self) -> T_co:
        """Backward compatibility property - equivalent to value."""
        if self.is_failure:
            msg = "Attempted to access data on failed result"
            raise FlextResult._get_exceptions().ValidationError(
                message=msg, error_code="VALIDATION_ERROR"
            )
        # Use the returns backend to unwrap the value (same as .value)
        return self._result.unwrap()

    @property
    def value_or_none(self) -> T_co | None:
        """Get value or None if failed."""
        return self._result.unwrap() if self.is_success else None

    def expect(self, message: str) -> T_co:
        """Get value or raise with custom message."""
        if self.is_failure:
            msg = f"{message}: {self._error}"
            raise FlextResult._get_exceptions().BaseError(
                message=msg, error_code="OPERATION_ERROR"
            )
        # DEFENSIVE: .expect() validates None for safety (unlike .value/.unwrap)
        if self._data is None:
            msg = "Success result has None data"
            raise FlextResult._get_exceptions().BaseError(
                message=msg, error_code="OPERATION_ERROR"
            )
        return self._data

    @override
    def __eq__(self, other: object) -> bool:
        """Check equality with another result using Python 3.13+ type narrowing."""
        if not isinstance(other, FlextResult):
            return False

        try:
            # Direct comparison with explicit type handling for Python 3.13+
            # Use explicit type annotations to help the type checker
            # Cast to object to avoid generic type issues
            self_data_obj: object = cast(
                "object",
                getattr(cast("object", self), "_data", None),
            )
            other_data_obj: object = cast(
                "object",
                getattr(cast("object", other), "_data", None),
            )

            # Avoid direct comparison of generic types by using identity check first
            if self_data_obj is other_data_obj:
                data_equal: bool = True
            else:
                # Use a more explicit approach to avoid type checker issues
                # Convert to string representation for comparison to avoid generic type issues
                try:
                    self_data_str: str = str(self_data_obj)
                    other_data_str: str = str(other_data_obj)
                    data_equal = self_data_str == other_data_str
                except (TypeError, ValueError, AttributeError):
                    # User data __str__() or comparison may raise exceptions
                    data_equal = False

            error_equal: bool = bool(self._error == other._error)
            code_equal: bool = bool(self._error_code == other._error_code)
            data_dict_equal: bool = bool(self._error_data == other._error_data)

            return data_equal and error_equal and code_equal and data_dict_equal
        except Exception:
            # VALIDATION HIERARCHY - User data comparison (HIGH PRIORITY)
            # User data __eq__(), __str__() methods can raise any exception
            # Level 1: Known exceptions (TypeError, ValueError, AttributeError, KeyError)
            # Level 2: User data custom exceptions (__eq__ can raise anything)
            # Safe default: return False if any exception during comparison
            return False

    @override
    def __hash__(self) -> int:
        """Return hash for a result to enable use in sets and dicts."""
        # Hash based on success state and primary content
        if self.is_success:
            # For success, hash the data (if hashable) or use a default
            try:
                return hash((True, self._data))
            except TypeError:
                # Data is not hashable - use alternative hashing strategy
                # Proper handling of non-hashable data
                # Use type-safe approach based on data characteristics
                if hasattr(self._data, "__dict__"):
                    # For objects with __dict__, hash their attributes
                    try:
                        attrs = tuple(sorted(self._data.__dict__.items()))
                        return hash((True, attrs))
                    except (TypeError, ValueError):
                        # Attributes not hashable or sortable, use fallback
                        # Unable to hash object attributes, using type+id fallback
                        return hash((
                            True,
                            type(self._data).__name__,
                            id(self._data),
                        ))
                    except Exception:
                        # VALIDATION HIERARCHY - User data hashing (HIGH PRIORITY)
                        # Accessing/sorting user __dict__ attributes can raise any exception
                        # Safe fallback: use type name and object identity
                        return hash((True, type(self._data).__name__, id(self._data)))

                # For complex objects, use a combination of type and memory ID
                return hash((True, type(self._data).__name__, id(self._data)))
            except Exception:
                # VALIDATION HIERARCHY - User data hashing (HIGH PRIORITY)
                # Hashing any user data can raise unexpected exceptions
                # Safe fallback: use type name and object identity
                return hash((True, type(self._data).__name__, id(self._data)))
        else:
            # For failure, hash the error message and code
            return hash((False, self._error, self._error_code))

    @override
    def __repr__(self) -> str:
        """Return string representation for debugging."""
        if self.is_success:
            return f"FlextResult(data={self._data!r}, is_success=True, error=None)"
        return f"FlextResult(data=None, is_success=False, error={self._error!r})"

    # Methods for a railway pattern

    def or_else(self, alternative: FlextResult[T_co]) -> FlextResult[T_co]:
        """Return this result if successful, otherwise return an alternative result."""
        if self.is_success:
            return self
        return alternative

    def or_else_get(self, func: Callable[[], FlextResult[T_co]]) -> Self:
        """Return this result if successful, otherwise return result of func."""
        if self.is_success:
            return self
        try:
            return cast("Self", func())
        except Exception as e:
            # VALIDATION HIERARCHY - User callable exception handling (CRITICAL)
            # Level 1: Known internal exceptions (TypeError, AttributeError)
            # Level 2: Common application exceptions (ValueError, RuntimeError)
            # Level 3: Catch-all for custom exceptions from user functions
            return cast("Self", FlextResult[T_co].fail(str(e)))

    def unwrap_or(self, default: T_co) -> T_co:
        """Return value or default if failed."""
        if self.is_success:
            return cast("T_co", self._data)
        return default

    def unwrap(self) -> T_co:
        """Get value or raise if failed."""
        if self.is_success:
            return cast("T_co", self._data)
        error_msg = self._error or "Operation failed"
        raise FlextResult._get_exceptions().BaseError(
            message=error_msg, error_code="OPERATION_ERROR"
        )

    def recover(self, func: Callable[[str], T_co]) -> FlextResult[T_co]:
        """Recover from failure by applying func to error.

        Note: If the recovery function raises an exception, it is re-raised.
        Recovery functions that might fail should handle their own exceptions.
        """
        if self.is_success:
            return self
        if self._error is not None:
            recovered_data: T_co = func(self._error)
            return FlextResult[T_co].ok(recovered_data)
        return FlextResult[T_co].fail("No error to recover from")

    def tap(self, func: Callable[[T_co], None]) -> FlextResult[T_co]:
        """Execute side effect function on success with non-None data, return self.

        Note: Side effect functions should be designed to not raise exceptions.
        If the function raises an exception, it indicates a programming error
        and should be fixed in the caller, not suppressed here.
        """
        if self.is_success and self._data is not None:
            func(self._data)
        return self

    # =========================================================================
    # ADDITIONAL RAILWAY METHODS - Enhanced error handling patterns
    # =========================================================================

    def lash(self, func: Callable[[str], FlextResult[T_co]]) -> FlextResult[T_co]:
        """Apply function to error value - opposite of flat_map (returns library pattern).

        This is the error-handling counterpart to flat_map. While flat_map operates
        on successful values, lash operates on error messages. This pattern is from
        the returns library and enables railway-oriented error recovery.

        Args:
            func: Function that takes error message and returns FlextResult[T_co]

        Returns:
            Self if success, otherwise result of applying func to error

        Example:
            ```python
            from flext_core import FlextResult


            def retry_on_network_error(error: str) -> FlextResult[dict]:
                if "network" in error.lower():
                    # Retry the operation
                    return FlextResult[dict].ok({"retried": True})
                # Pass through other errors
                return FlextResult[dict].fail(error)


            # Success case - lash not applied
            result = FlextResult[dict].ok({"data": "value"})
            final = result.lash(retry_on_network_error)
            # final is Success({"data": "value"})

            # Network failure - lash triggers retry
            result_fail = FlextResult[dict].fail("Network timeout error")
            final_retry = result_fail.lash(retry_on_network_error)
            # final_retry is Success({"retried": True})

            # Other failure - lash passes through
            result_other = FlextResult[dict].fail("Validation error")
            final_other = result_other.lash(retry_on_network_error)
            # final_other is Failure("Validation error")
            ```

        """
        if self.is_success:
            return self

        error_msg = self._error or ""
        try:
            return func(error_msg)
        except Exception as e:
            # VALIDATION HIERARCHY - User callable exception handling (CRITICAL)
            # Level 1: Known internal exceptions (TypeError, AttributeError)
            # Level 2: Common application exceptions (ValueError, RuntimeError)
            # Level 3: Catch-all for custom exceptions from user functions
            return FlextResult[T_co].fail(f"Lash operation failed: {e}")

    def alt(self, default_result: FlextResult[T_co]) -> FlextResult[T_co]:
        """Return self if success, otherwise default_result (alias for or_else from returns).

        This method provides an alternative result when the current result fails.
        It's an alias for or_else but uses the returns library naming convention.

        Args:
            default_result: Alternative FlextResult to use if this one failed

        Returns:
            Self if success, otherwise default_result

        Example:
            ```python
            from flext_core import FlextResult


            # Success case - alt not used
            result = FlextResult[int].ok(42)
            default = FlextResult[int].ok(0)
            final = result.alt(default)
            # final is Success(42)

            # Failure case - alt provides fallback
            result_fail = FlextResult[int].fail("Primary source failed")
            default_ok = FlextResult[int].ok(0)
            final_fallback = result_fail.alt(default_ok)
            # final_fallback is Success(0)

            # Chain multiple alternatives
            primary = FlextResult[int].fail("Primary failed")
            secondary = FlextResult[int].fail("Secondary failed")
            tertiary = FlextResult[int].ok(999)

            final_chain = primary.alt(secondary).alt(tertiary)
            # final_chain is Success(999)
            ```

        """
        return self if self.is_success else default_result

    def value_or_call(self, func: Callable[[], T_co]) -> T_co:
        """Get value or call func to get default (lazy evaluation pattern).

        Unlike unwrap_or which requires the default value upfront, this method
        takes a callable that's only executed if the result is a failure. This
        enables lazy evaluation of expensive default values.

        Args:
            func: Callable that returns default value (only called if failure)

        Returns:
            Value if success, otherwise result of calling func

        Example:
            ```python
            from flext_core import FlextResult


            def expensive_default() -> dict[str, object]:
                # This only runs if result is failure
                print("Computing expensive default...")
                return {"default": True, "computed": True}


            # Success case - func not called
            result = FlextResult[dict].ok({"data": "value"})
            value = result.value_or_call(expensive_default)
            # value is {"data": "value"}, expensive_default never called

            # Failure case - func called lazily
            result_fail = FlextResult[dict].fail("Error")
            value_default = result_fail.value_or_call(expensive_default)
            # Prints: Computing expensive default...
            # value_default is {"default": True, "computed": True}

            # Use with lambda for inline defaults
            result = FlextResult[int].fail("Error")
            value = result.value_or_call(lambda: 42)
            # value is 42
            ```

        """
        if self.is_success:
            return cast("T_co", self._data)

        try:
            return func()
        except Exception as e:
            # If default computation fails, we need to handle it somehow
            # Since this returns T_co not FlextResult, we raise
            error_message = f"Default value computation failed: {e}"
            raise FlextResult._get_exceptions().BaseError(
                message=error_message, error_code="OPERATION_ERROR"
            ) from e

    def filter(
        self,
        predicate: Callable[[T_co], bool],
        error_msg: str = "Filter predicate failed",
    ) -> FlextResult[T_co]:
        """Filter success value with predicate."""
        if self.is_failure:
            return self
        try:
            # Apply predicate using discriminated union type narrowing
            # Python 3.13+ discriminated union: _data is guaranteed to be T_co for success
            if predicate(cast("T_co", self._data)):
                return self
            return FlextResult[T_co].fail(error_msg)
        except Exception as e:
            return FlextResult[T_co].fail(str(e))

    @classmethod
    def from_exception(
        cls,
        func: Callable[[], object],
    ) -> FlextResult[object]:
        """Create a result from a function that might raise exception."""
        try:
            result = func()
            return FlextResult[object].ok(result)
        except Exception as e:
            return FlextResult[object].fail(str(e))

    # =========================================================================
    # MAYBE INTEROP - Convert between FlextResult and returns.maybe.Maybe
    # =========================================================================

    def to_maybe(self) -> Some[T_co | None] | Maybe[Never]:
        """Convert FlextResult to returns.maybe.Maybe for optional value semantics.

        This enables interoperability with the returns library Maybe monad,
        which represents optional values (Some/Nothing) without error messages.

        Returns:
            Maybe[T_co]: Some(value) if success, Nothing if failure

        Example:
            ```python
            from flext_core import FlextResult


            # Success case converts to Some
            result = FlextResult[int].ok(42)
            maybe = result.to_maybe()
            # maybe is Some(42)

            # Failure case converts to Nothing
            result = FlextResult[int].fail("Error occurred")
            maybe = result.to_maybe()
            # maybe is Nothing (error message is lost)

            # Use with returns library operations
            from returns.pipeline import flow
            from returns.pointfree import map_

            result = FlextResult[int].ok(5)
            doubled = flow(
                result.to_maybe(),
                map_(lambda x: x * 2),  # Pointfree map
            )
            # doubled is Some(10)
            ```

        """
        return Some[T_co | None](self._data) if self.is_success else Nothing

    @classmethod
    def from_maybe(cls, maybe: object) -> FlextResult[object]:
        """Create FlextResult from returns.maybe.Maybe.

        Converts a Maybe monad from the returns library into a FlextResult,
        preserving the value if Some or creating a failure if Nothing.

        Args:
            maybe: Maybe[T] from returns library

        Returns:
            FlextResult[T]: Success with value if Some, Failure if Nothing

        Example:
            ```python
            from flext_core import FlextResult
            from returns.maybe import Maybe, Some, Nothing


            # Convert Some to success
            maybe_value = Some(42)
            result = FlextResult[int].from_maybe(maybe_value)
            assert result.is_success
            assert result.value == 42

            # Convert Nothing to failure
            maybe_nothing = Nothing
            result = FlextResult[int].from_maybe(maybe_nothing)
            assert result.is_failure
            assert result.error == "No value in Maybe"

            # Pipeline example
            from returns.maybe import maybe


            @maybe
            def find_user(user_id: int) -> int | None:
                return user_id if user_id > 0 else None


            # Convert Maybe result to FlextResult
            result = FlextResult[int].from_maybe(find_user(5))
            # result is Success(5)
            ```

        """
        # Check if it's Some (has a value)
        if isinstance(maybe, Some):
            # Extract value using unwrap() - safely typed extraction
            # The cast is safe because isinstance check guarantees Some type
            value = maybe.unwrap()
            return FlextResult[object].ok(value)

        # It's Nothing or extraction failed
        return FlextResult[object].fail("No value in Maybe")

    @classmethod
    def sequence[T](cls, results: list[FlextResult[T]]) -> FlextResult[list[T]]:
        """Convert list of results to result of list, failing on first failure.

        Args:
            results: List of FlextResult instances to sequence

        Returns:
            FlextResult containing list of all values if all successful,
            or first failure encountered.

        """
        return FlextResult._sequence_results(results)

    # =========================================================================
    # UTILITY METHODS - formerly FlextResultUtils
    # =========================================================================

    @classmethod
    def collect_successes[TCollect](
        cls, results: list[FlextResult[TCollect]]
    ) -> list[TCollect]:
        """Collect successful values from results."""
        return [result.value for result in results if result.is_success]

    @classmethod
    def collect_failures[TCollectFail](
        cls,
        results: list[FlextResult[TCollectFail]],
    ) -> list[str]:
        """Collect error messages from failures."""
        return [r.error for r in results if r.is_failure and r.error]

    @classmethod
    def success_rate[TUtil](cls, results: list[FlextResult[TUtil]]) -> float:
        """Calculate success rate percentage."""
        if not results:
            return 0.0
        successes = sum(1 for r in results if r.is_success)
        return (successes / len(results)) * 100.0

    @classmethod
    def batch_process[TBatch, UBatch](
        cls,
        items: list[TBatch],
        processor: Callable[[TBatch], FlextResult[UBatch]],
    ) -> tuple[list[UBatch], list[str]]:
        """Process batch and separate successes from failures."""
        results: list[FlextResult[UBatch]] = [processor(item) for item in items]
        successes: list[UBatch] = cls.collect_successes(results)
        failures: list[str] = cls.collect_failures(results)
        return successes, failures

    @classmethod
    def safe_call[TResult](
        cls: type[FlextResult[TResult]],
        func: Callable[[], TResult],
        *,
        error_code: str | None = None,
    ) -> FlextResult[TResult]:
        """Execute function safely, wrapping result in FlextResult.

        Similar to dry-python/returns @safe decorator but as a classmethod
        for inline use. Catches all exceptions and converts them to
        FlextResult failures with optional error code.

        Args:
            func: Callable that returns TResult
            error_code: Optional error code for failures (defaults to OPERATION_ERROR)

        Returns:
            FlextResult[TResult] wrapping the function result or error

        Example:
            ```python
            def fetch_data() -> dict[str, object]:
                return api.get_data()


            # Basic usage
            result = FlextResult["dict[str, object]"].safe_call(fetch_data)

            # With error code
            result = FlextResult["dict[str, object]"].safe_call(
                fetch_data, error_code="API_ERROR"
            )

            if result.is_success:
                data = result.unwrap()
            ```

        """
        try:
            value = func()
            return FlextResult[TResult].ok(value)
        except Exception as e:
            return FlextResult[TResult].fail(
                str(e),
                error_code=error_code or FlextConstants.Errors.OPERATION_ERROR,
            )

    # === MONADIC COMPOSITION ADVANCED OPERATORS (Python 3.13) ===

    def __rshift__[U](self, func: Callable[[T_co], FlextResult[U]]) -> FlextResult[U]:
        """Right shift operator (>>) for monadic bind - ADVANCED COMPOSITION."""
        return self.flat_map(func)

    def __lshift__[U](self, func: Callable[[T_co], U]) -> FlextResult[U]:
        """Left shift operator (<<) for functor map - ADVANCED COMPOSITION."""
        return self.map(func)

    def __mod__(self, predicate: Callable[[T_co], bool]) -> FlextResult[T_co]:
        """Modulo operator (%) for conditional filtering - ADVANCED COMPOSITION."""
        if self.is_failure:
            return self

        try:
            if predicate(self.unwrap()):
                return self
            return FlextResult[T_co].fail(
                f"{FlextConstants.Messages.VALIDATION_FAILED} (predicate)",
            )
        except Exception as e:
            return FlextResult[T_co].fail(f"Predicate evaluation failed: {e}")

    def __xor__(self, recovery_func: Callable[[str], T_co]) -> FlextResult[T_co]:
        """XOR operator (^) for error recovery - ADVANCED COMPOSITION."""
        return self.recover(recovery_func)

    # === ADVANCED MONADIC COMBINATORS (Category Theory) ===

    @classmethod
    def traverse[TTraverse, UTraverse](
        cls,
        items: list[TTraverse],
        func: Callable[[TTraverse], FlextResult[UTraverse]],
    ) -> FlextResult[list[UTraverse]]:
        """Traverse a list with a function returning FlextResults."""
        results: list[UTraverse] = []
        for item in items:
            result: FlextResult[UTraverse] = func(item)
            if result.is_failure:
                return FlextResult[list[UTraverse]].fail(
                    result.error or f"Traverse failed at item {item}",
                )
            results.append(result.unwrap())
        return FlextResult[list[UTraverse]].ok(results)

    # === RAILWAY-ORIENTED PROGRAMMING ENHANCEMENTS ===

    @classmethod
    def pipeline[TPipeline](
        cls,
        initial_value: TPipeline,
        *operations: Callable[[TPipeline], FlextResult[TPipeline]],
    ) -> FlextResult[TPipeline]:
        """Compose multiple operations into a single pipeline.

        Args:
            initial_value: Starting value for the pipeline
            *operations: Operations to chain together

        Returns:
            Final result after all operations or first failure

        """
        current_result: FlextResult[TPipeline] = FlextResult[TPipeline].ok(
            initial_value,
        )

        for operation in operations:
            current_result = current_result.flat_map(operation)
            if current_result.is_failure:
                break

        return current_result

    def with_context(self, context_func: Callable[[str], str]) -> FlextResult[T_co]:
        """Add contextual information to error messages.

        Args:
            context_func: Function to transform error message with context

        Returns:
            Same result with enhanced error context if failed

        """
        if self.is_success:
            return self

        if self.error:
            enhanced_error = context_func(self.error)
            return FlextResult[T_co].fail(
                enhanced_error,
                error_code=self.error_code,
                error_data=self.error_data,
            )
        return self

    # === NEW ADVANCED MONADIC OPERATORS FOR COMPLEXITY REDUCTION ===

    @classmethod
    def accumulate_errors[TAccumulate](
        cls,
        *results: FlextResult[TAccumulate],
    ) -> FlextResult[list[TAccumulate]]:
        """Accumulate all errors or return all successes."""
        successes: list[TAccumulate] = []
        errors: list[str] = []

        for result in results:
            if result.is_success:
                successes.append(result.unwrap())
            else:
                errors.append(result.error or "Unknown error")

        if errors:
            combined_error = "; ".join(errors)
            return FlextResult[list[TAccumulate]].fail(
                f"Multiple errors occurred: {combined_error}",
                error_code="ACCUMULATED_ERRORS",
                error_data={"error_count": len(errors), "errors": errors},
            )

        return FlextResult[list[TAccumulate]].ok(successes)

    @classmethod
    def parallel_map[TPar, UPar](
        cls,
        items: list[TPar],
        func: Callable[[TPar], FlextResult[UPar]],
        *,
        fail_fast: bool = True,
    ) -> FlextResult[list[UPar]]:
        """Map function over items in parallel (conceptually)."""
        results: list[FlextResult[UPar]] = [func(item) for item in items]

        if fail_fast:
            fast_successes: list[UPar] = []
            for result in results:
                if result.is_failure:
                    error_msg = result.error or "Sequence operation failed"
                    return FlextResult[list[UPar]].fail(error_msg)
                fast_successes.append(result.unwrap())
            return FlextResult[list[UPar]].ok(fast_successes)

        successes: list[UPar] = []
        errors: list[str] = []
        for result in results:
            if result.is_failure:
                error_msg = result.error or "Operation failed"
                errors.append(error_msg)
            else:
                successes.append(result.unwrap())

        if errors:
            return FlextResult[list[UPar]].fail("; ".join(errors))
        return FlextResult[list[UPar]].ok(successes)

    def with_resource[TResource, UResource](
        self,
        resource_factory: Callable[[], TResource],
        operation: Callable[[T_co, TResource], FlextResult[UResource]],
        cleanup: Callable[[TResource], None] | None = None,
    ) -> FlextResult[UResource]:
        """Execute operation with automatic resource management.

        Args:
            resource_factory: Function to create the resource
            operation: Function to execute with value and resource
            cleanup: Optional cleanup function for the resource

        Returns:
            Result of operation with guaranteed resource cleanup

        """
        if self.is_failure:
            return FlextResult[UResource].fail(
                self.error or "Cannot use resource with failed result",
                error_code=self.error_code,
                error_data=self.error_data,
            )

        try:
            resource = resource_factory()
            try:
                return operation(self.unwrap(), resource)
            finally:
                if cleanup:
                    cleanup(resource)
        except Exception as e:
            return FlextResult[UResource].fail(f"Resource operation failed: {e}")

    # =========================================================================
    # UTILITY METHODS - Moved from nested classes to main class
    # =========================================================================

    @staticmethod
    def _combine_results[TCombine](
        results: list[FlextResult[TCombine]],
    ) -> FlextResult[list[TCombine]]:
        """Combine multiple results into a single result."""
        values: list[TCombine] = []
        for result in results:
            if result.is_failure:
                return FlextResult[list[TCombine]].fail(
                    result.error or "Combine operation failed",
                )
            values.append(result.value)
        return FlextResult[list[TCombine]].ok(values)

    @classmethod
    def _sequence_results[TSeq](
        cls,
        results: list[FlextResult[TSeq]],
    ) -> FlextResult[list[TSeq]]:
        """Sequence a list of results into a result of list."""
        values: list[TSeq] = []
        for result in results:
            if result.is_failure:
                return FlextResult[list[TSeq]].fail(
                    result.error or "Sequence failed",
                    error_code=result.error_code,
                    error_data=result.error_data,
                )
            values.append(result.unwrap())
        return FlextResult[list[TSeq]].ok(values)

    @classmethod
    def validate_all[TValidateAll](
        cls,
        value: TValidateAll,
        *validators: Callable[[TValidateAll], FlextResult[None]],
    ) -> FlextResult[TValidateAll]:
        """Validate data with multiple validators."""
        validation_results: list[FlextResult[None]] = [
            validator(value) for validator in validators
        ]
        errors = [
            result.error
            for result in validation_results
            if result.is_failure and result.error
        ]

        if errors:
            combined_error = "; ".join(errors)
            return FlextResult[TValidateAll].fail(
                f"{FlextConstants.Messages.VALIDATION_FAILED}: {combined_error}",
                error_code="VALIDATION_FAILED",
                error_data={
                    "validation_errors": errors,
                    "error_count": len(errors),
                },
            )

        return FlextResult[TValidateAll].ok(value)

    @staticmethod
    def collect_all_errors[TCollectErr](
        *results: FlextResult[TCollectErr],
    ) -> tuple[list[TCollectErr], list[str]]:
        """Collect all successful values and error messages from results.

        Args:
            *results: Variable number of FlextResult instances to process.

        Returns:
            Tuple of (successes, errors) where:
            - successes: List of unwrapped values from successful results
            - errors: List of error messages from failed results

        Example:
            >>> results = [
            ...     FlextResult[int].ok(1),
            ...     FlextResult[int].fail("Error 1"),
            ...     FlextResult[int].ok(2),
            ... ]
            >>> successes, errors = FlextResult.collect_all_errors(*results)
            >>> print(successes)  # [1, 2]
            >>> print(errors)  # ["Error 1"]

        """
        successes: list[TCollectErr] = []
        errors: list[str] = []

        for result in results:
            if result.is_success:
                successes.append(result.value)
            elif result.error:
                errors.append(result.error)

        return successes, errors

    @staticmethod
    def map_sequence[TMapSeq, UMapSeq](
        items: list[TMapSeq],
        mapper: Callable[[TMapSeq], FlextResult[UMapSeq]],
    ) -> FlextResult[list[UMapSeq]]:
        """Map a function over a sequence, failing on first error.

        Applies the mapper function to each item in the sequence,
        collecting successful results. Returns failure on the first
        error encountered (fail-fast behavior).

        Args:
            items: List of items to map over.
            mapper: Function that takes an item and returns a FlextResult.

        Returns:
            FlextResult containing list of mapped values, or failure
            with the first error encountered.

        Example:
            >>> def double(x: int) -> FlextResult[int]:
            ...     if x < 0:
            ...         return FlextResult[int].fail("Negative number")
            ...     return FlextResult[int].ok(x * 2)
            >>>
            >>> result = FlextResult.map_sequence([1, 2, 3], double)
            >>> print(result.value)  # [2, 4, 6]
            >>>
            >>> result = FlextResult.map_sequence([1, -1, 3], double)
            >>> print(result.error)  # "Negative number"

        """
        mapped: list[UMapSeq] = []

        for item in items:
            result = mapper(item)
            if result.is_failure:
                return FlextResult[list[UMapSeq]].fail(
                    result.error or "Mapping failed",
                    error_code=result.error_code,
                    error_data=result.error_data,
                )
            mapped.append(result.value)

        return FlextResult[list[UMapSeq]].ok(mapped)

    @staticmethod
    def _is_flattenable_sequence(item: object) -> bool:
        return isinstance(item, Sequence) and not isinstance(
            item,
            (str, bytes, bytearray),
        )

    @staticmethod
    def _flatten_variadic_args(*items: object) -> list[object]:
        flat: list[object] = []
        for item in items:
            if FlextResult._is_flattenable_sequence(item) and isinstance(
                item,
                (list, tuple),
            ):
                # Cast item to the expected type for recursive call
                item_cast = cast("tuple[object, ...]", item)
                flat.extend(FlextResult._flatten_variadic_args(*item_cast))
            else:
                flat.append(item)
        return flat

    @staticmethod
    def _flatten_callable_args(*items: object) -> list[Callable[[], object]]:
        flat_callables: list[Callable[[], object]] = []
        for item in items:
            if FlextResult._is_flattenable_sequence(item) and isinstance(
                item,
                (list, tuple),
            ):
                # Cast item to the expected type for recursive call
                item_cast = cast("tuple[object, ...]", item)
                flat_callables.extend(FlextResult._flatten_callable_args(*item_cast))
            else:
                if not callable(item):
                    msg = "Expected callable when flattening alternatives"
                    raise FlextResult._get_exceptions().ValidationError(
                        message=msg, error_code="VALIDATION_ERROR"
                    )
                flat_callables.append(cast("Callable[[], object]", item))
        return flat_callables

    def validate_and_execute[T, U](
        self,
        validator: Callable[[T_co], FlextResult[None]],
        executor: Callable[[T_co], FlextResult[U]],
    ) -> FlextResult[U]:
        """Validate and execute operations on successful results.

        Args:
            validator: Function to validate the result value
            executor: Function to execute if validation passes

        Returns:
            FlextResult with execution result or validation/execution failure

        """
        if self.is_failure:
            return FlextResult[U].fail(
                self.error or "Validation failed",
                error_code=self.error_code,
                error_data=self.error_data,
            )

        # Validate the current value
        validation_result = validator(self.value)
        if validation_result.is_failure:
            return FlextResult[U].fail(
                validation_result.error or "Validation failed",
                error_code=validation_result.error_code,
                error_data=validation_result.error_data,
            )

        # Execute if validation passed
        try:
            return executor(self.value)
        except Exception as e:
            return FlextResult[U].fail(f"Execution failed: {e}")

    @staticmethod
    def chain_validations[T](
        *validators: Callable[[], FlextResult[T]],
    ) -> FlextResult[T]:
        """Chain multiple validation functions together.

        Args:
            *validators: Validation functions to chain

        Returns:
            Function that applies all validations in sequence

        """
        if not validators:
            return FlextResult[T].fail("No validators provided")

        # Execute validators in sequence
        result = validators[0]()
        for validator in validators[1:]:
            if result.is_failure:
                break
            result = validator()

        return result

    # =========================================================================
    # IO INTEROP - Convert between FlextResult and returns.io types
    # =========================================================================

    def to_io(self) -> object:
        """Wrap success value in returns.io.IO for pure side effects."""
        if self.is_failure:
            error_msg = self._error or "Failed"
            msg = f"Cannot convert failure to IO: {error_msg}"
            raise FlextResult._get_exceptions().ValidationError(
                message=msg, error_code="VALIDATION_ERROR"
            )
        return IO(self._data)

    def to_io_result(self) -> IOResult[object, object]:
        """Convert FlextResult to returns.io.IOResult for impure operations."""
        if self.is_success:
            return IOSuccess(self._data)
        error_msg = self._error or "Operation failed"
        return IOFailure(error_msg)

    @classmethod
    def from_io_result[T](
        cls: type[FlextResult[T]], io_result: IOResult[object, object]
    ) -> FlextResult[T]:
        """Create FlextResult from returns.io.IOResult using public API.

        IOResult behavior:
        - IOSuccess(value).map(f) → f(value) is called, extracts success value
        - IOFailure(error).alt(f) → f(error) is called, extracts error value
        Only ONE of the two callbacks executes based on success/failure state.
        """
        # Track which path was taken using separate lists
        extracted_success: list[object] = []
        extracted_failure: list[object] = []

        def extract_success(result: object) -> object:
            """Extract value from IOSuccess via map()."""
            extracted_success.append(result)
            return result

        def extract_failure(result: object) -> object:
            """Extract error from IOFailure via alt()."""
            extracted_failure.append(result)
            return result

        # Chain map() and alt() - only ONE callback executes
        # IOSuccess and IOFailure both inherit from IOResult which has both map() and alt()
        io_result.map(extract_success).alt(extract_failure)

        # Check success path (IOSuccess.map() was called)
        if extracted_success:
            value = extracted_success[0]

            # Handle Success[T] wrapper (uncommon, but possible)
            if isinstance(value, Success):
                value = value.unwrap()

            # Return success with extracted value
            return cls(data=cast("T", value))

        # Check failure path (IOFailure.alt() was called)
        if extracted_failure:
            error_value = extracted_failure[0]

            # Handle Failure wrapper (uncommon, but possible)
            if isinstance(error_value, Failure):
                error_value = error_value.failure()

            # Convert error to string
            error_msg = (
                str(error_value) if error_value is not None else "Operation failed"
            )
            return cls(error=error_msg)

        # Neither callback executed - unexpected IOResult state
        return cls(error="Failed to extract from IOResult")


__all__ = [
    "FlextResult",
]
