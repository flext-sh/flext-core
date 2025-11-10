"""Base classes for CQRS command and query handlers.

This module provides FlextHandlers, the base class for implementing
Command Query Responsibility Segregation (CQRS) handlers throughout
the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import inspect
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from typing import (
    ClassVar,
    Literal,
    Self,
    cast,
    override,
)

from beartype.door import is_bearable
from pydantic import BaseModel

from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import (
    CallableInputT,
    CallableOutputT,
    FlextTypes,
)
from flext_core.utilities import FlextUtilities


class FlextHandlers[MessageT_contra, ResultT](FlextMixins, ABC):
    """Base class for CQRS command and query handlers.

    Implements FlextProtocols.Handler[MessageT_contra] through structural typing.
    All handler subclasses automatically satisfy the Handler protocol by
    implementing the required methods: handle(), validate(), __call__(),
    can_handle(), execute(), validate_command(), and validate_query().

    Provides the foundation for implementing CQRS handlers with validation,
    execution context, metrics collection, and configuration management.
    Supports commands, queries, events, and sagas with type safety.

    Protocol Compliance:
        ✅ STRUCTURAL TYPING: Implements FlextProtocols.Handler[MessageT_contra]

        Implemented Protocol Methods:
        - handle(message: MessageT_contra) -> object - Abstract method for subclasses
        - execute(message: MessageT_contra) -> object - Execute handler with message
        - __call__(input_data: MessageT_contra) -> object - Callable interface
        - validate(data: FlextTypes.AcceptableMessageType) -> FlextResult[None] - Validate input
        - validate_command(command: FlextTypes.AcceptableMessageType) -> FlextResult[None] - Validate command
        - validate_query(query: FlextTypes.AcceptableMessageType) -> FlextResult[None] - Validate query
        - can_handle(message_type: object) -> bool - Check handler capability

    Features:
    - Abstract base for command/query/event handlers
    - Handler execution with validation pipeline
    - Type checking for message compatibility
    - Metrics collection for handler performance
    - Configuration via handler models
    - Execution context tracking per handler
    - Message validation with FlextResult
    - Logger integration for handler operations
    - Automatic protocol satisfaction via structural typing

    Usage:
        >>> from flext_core import FlextHandlers, FlextResult
        >>> from flext_core.protocols import FlextProtocols
        >>>
        >>> class CreateUserHandler(FlextHandlers[CreateUserCommand, User]):
        ...     def handle(self, command: CreateUserCommand) -> FlextResult[User]:
        ...         return self.ok(User(name=command.name))
        >>>
        >>> handler = CreateUserHandler(config=...)
        >>> # FlextHandlers explicitly implements FlextProtocols.Handler
        >>> assert isinstance(handler, FlextProtocols.Handler)  # ✅ Protocol satisfied
        >>> assert handler.can_handle(CreateUserCommand)  # ✅ Handler validation
        >>> result = handler.execute(CreateUserCommand(name="Alice"))  # ✅ Execution
    """

    # Class-level logger for internal operations (not for subclass use)
    _internal_logger: FlextLogger = FlextLogger(__name__)

    # Runtime type validation attributes
    _expected_message_type: ClassVar[type | None] = None
    _expected_result_type: ClassVar[type | None] = None

    # Type parameter count for generic subscription
    _REQUIRED_TYPE_PARAM_COUNT: ClassVar[int] = 2

    def __class_getitem__(cls, item: tuple[type, type] | type) -> type[Self]:
        """Intercept FlextHandlers[M, R] to create typed subclass with validation.

        This enables automatic runtime type validation for handlers:
        - FlextHandlers[CreateUserCommand, User] validates message and result types
        - FlextHandlers[Query, Report] validates both input and output types

        Args:
            item: Either tuple of (MessageType, ResultType) or single ResultType

        Returns:
            Typed subclass with expected types stored as class variables

        Examples:
            >>> class UserHandler(FlextHandlers[CreateUserCommand, User]):
            ...     def handle(self, msg: CreateUserCommand) -> FlextResult[User]:
            ...         return self.ok(User(name=msg.name))
            >>>
            >>> # ✅ Correct types - passes validation
            >>> handler = UserHandler(config=...)
            >>> result = handler.execute(CreateUserCommand(name="Alice"))
            >>>
            >>> # ❌ Wrong result type - automatic rejection
            >>> class BadHandler(FlextHandlers[CreateUserCommand, User]):
            ...     def handle(self, msg: CreateUserCommand) -> FlextResult[User]:
            ...         return self.ok(Product(id="123"))  # Wrong type!

        """
        # Handle both single and tuple types
        message_type: type | None
        result_type: type

        if isinstance(item, tuple):
            if len(item) != cls._REQUIRED_TYPE_PARAM_COUNT:
                msg = "FlextHandlers requires exactly 2 type parameters: FlextHandlers[MessageType, ResultType]"
                raise TypeError(msg)
            message_type, result_type = item
        else:
            # Single type parameter - assume it's result type, message type is Any
            message_type = None
            result_type = item

        # Create typed subclass dynamically using type() built-in
        cls_name = getattr(cls, "__name__", "FlextHandlers")
        cls_qualname = getattr(cls, "__qualname__", "FlextHandlers")
        msg_name = getattr(message_type, "__name__", "Any") if message_type else "Any"
        res_name = getattr(result_type, "__name__", str(result_type))

        # Create typed subclass dynamically using type() built-in
        # Type checkers cannot verify dynamic type() calls with 3 arguments
        # This is valid Python metaprogramming - dynamically creating a class at runtime
        typed_subclass: type[Self] = type(  # type: ignore[call-overload]
            f"{cls_name}[{msg_name}, {res_name}]",
            (cls,),
            {
                "_expected_message_type": message_type,
                "_expected_result_type": result_type,
            },
        )

        # Preserve qualname for better debugging
        typed_subclass.__qualname__ = f"{cls_qualname}[{msg_name}, {res_name}]"

        return typed_subclass

    class _MessageValidator:
        """Private message validation utilities for FlextHandlers."""

        _SERIALIZABLE_MESSAGE_EXPECTATION = (
            "dict, str, int, float, bool, dataclass, attrs class, or object exposing "
            "model_dump/dict/as_dict/__slots__ representations"
        )

        @classmethod
        def validate_message(
            cls,
            message: object,
            *,
            operation: str,
            revalidate_pydantic_messages: bool = False,
        ) -> FlextResult[None]:
            """Validate a message for the given operation.

            Args:
                message: The message object to validate
                operation: The operation name for context
                revalidate_pydantic_messages: Whether to revalidate Pydantic models

            Returns:
                FlextResult[None]: Success if valid, failure with error details if invalid

            """
            # Check for custom validation methods first
            custom_validation = cls._validate_custom_method(message, operation)
            if custom_validation is not None:
                return custom_validation

            # Validate Pydantic models
            pydantic_validation = cls._validate_pydantic_model(
                message, operation, revalidate=revalidate_pydantic_messages
            )
            if pydantic_validation is not None:
                return pydantic_validation

            # Validate message serialization for non-Pydantic objects
            return cls._validate_message_serialization(message, operation)

        @classmethod
        def _validate_custom_method(
            cls,
            message: object,
            operation: str,
        ) -> FlextResult[None] | None:
            """Check for and execute custom validation method.

            Args:
                message: The message object to validate
                operation: The operation name

            Returns:
                FlextResult error if custom validation failed, None if no validation or success

            """
            validation_method_name = f"validate_{operation}"
            if not hasattr(message, validation_method_name):
                return None

            validation_method = getattr(message, validation_method_name)
            if not callable(validation_method):
                return None

            try:
                sig = inspect.signature(validation_method)
                if len(sig.parameters) != 0:
                    return None

                validation_result_obj = validation_method()
                if not isinstance(validation_result_obj, FlextResult):
                    return None

                validation_result: FlextResult[object] = cast(
                    "FlextResult[object]",
                    validation_result_obj,
                )
                if validation_result.is_failure:
                    return FlextResult[None].fail(
                        validation_result.error or f"{operation} validation failed",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )
            except Exception as e:
                # VALIDATION HIERARCHY - User data validation (HIGH)
                # Validation methods on user data can raise any exception
                # Safe behavior: log and continue with next validation
                FlextHandlers._internal_logger.debug(
                    f"Skipping validation method {validation_method_name}: {type(e).__name__}"
                )

            return None

        @classmethod
        def _validate_pydantic_model(
            cls,
            message: object,
            operation: str,
            *,
            revalidate: bool,
        ) -> FlextResult[None] | None:
            """Validate Pydantic models.

            Args:
                message: The message object
                operation: The operation name
                revalidate: Whether to revalidate Pydantic models

            Returns:
                FlextResult error if validation failed, None if not a Pydantic model

            """
            if not isinstance(message, BaseModel):
                return None

            if not revalidate:
                return FlextResult[None].ok(None)

            try:
                message.__class__.model_validate(message.model_dump(mode="python"))
                return FlextResult[None].ok(None)
            except Exception as e:
                # VALIDATION HIERARCHY - Pydantic model revalidation (HIGH)
                # User model_validate() can raise any exception
                # Safe behavior: capture validation error with full context
                validation_error = FlextExceptions.ValidationError(
                    f"Pydantic revalidation failed: {e}",
                    field="pydantic_model",
                    value=str(message)[: FlextConstants.Defaults.MAX_MESSAGE_LENGTH]
                    if hasattr(message, "__str__")
                    else "unknown",
                    correlation_id=f"pydantic_validation_{int(time.time() * 1000)}",
                    metadata={
                        "validation_details": f"pydantic_exception: {e!s}, model_class: {message.__class__.__name__}, revalidated: True",
                        "context": f"operation: {operation}, message_type: {type(message).__name__}, validation_type: pydantic_revalidation",
                    },
                )
                return FlextResult[None].fail(
                    str(validation_error),
                    error_code=validation_error.error_code,
                    error_data={"exception_context": str(validation_error)},
                )

        @classmethod
        def _validate_message_serialization(
            cls,
            message: object,
            operation: str,
        ) -> FlextResult[None]:
            """Validate message serialization for non-Pydantic objects.

            Args:
                message: The message object
                operation: The operation name

            Returns:
                FlextResult: Success if serializable, failure otherwise

            """
            try:
                cls._build_serializable_message_payload(message, operation=operation)
            except Exception as exc:
                # VALIDATION HIERARCHY - Message serialization (HIGH)
                # User message serialization can raise any exception
                # Safe behavior: check exception type and respond appropriately
                if isinstance(exc, FlextExceptions.TypeError):
                    return FlextResult[None].fail(
                        str(exc),
                        error_code=exc.error_code,
                        error_data={
                            "exception_context": getattr(exc, "context", str(exc))
                        },
                    )

                fallback_error = FlextExceptions.TypeError(
                    f"Invalid message type for {operation}: {type(message).__name__}",
                    expected_type=cls._SERIALIZABLE_MESSAGE_EXPECTATION,
                    actual_type=type(message).__name__,
                    context=f"operation: {operation}, message_type: {type(message).__name__}, validation_type: serializable_check, original_exception: {exc!s}",
                    correlation_id=f"type_validation_{int(time.time() * 1000)}",
                )
                return FlextResult[None].fail(
                    str(fallback_error),
                    error_code=fallback_error.error_code,
                    error_data={"exception_context": str(fallback_error)},
                )

            return FlextResult[None].ok(None)

        @classmethod
        def _build_serializable_message_payload(
            cls,
            message: object,
            *,
            operation: str | None = None,
        ) -> object:
            """Build a serializable representation for message validation heuristics."""
            operation_name = operation or "message"
            context_operation = operation or "unknown"

            # Try different message types in order
            if isinstance(message, (dict, str, int, float, bool)):
                return cast("dict[str, object] | str | int | float | bool", message)

            if message is None:
                cls._raise_invalid_message_type(
                    operation_name, context_operation, "NoneType"
                )

            if isinstance(message, BaseModel):
                return message.model_dump()

            if is_dataclass(message) and not isinstance(message, type):
                return asdict(message)

            # Handle attrs classes
            attrs_result = cls._try_attrs_serialization(message)
            if attrs_result is not None:
                return attrs_result

            # Try common serialization methods
            common_result = cls._try_common_serialization_methods(message)
            if common_result is not None:
                return common_result

            # Handle __slots__
            slots_result = cls._try_slots_serialization(message)
            if slots_result is not None:
                return slots_result

            if hasattr(message, "__dict__"):
                return vars(message)

            cls._raise_invalid_message_type(
                operation_name, context_operation, type(message).__name__
            )
            return None

        @classmethod
        def _try_attrs_serialization(cls, message: object) -> dict[str, object] | None:
            """Try to serialize attrs class."""
            attrs_fields = getattr(message, "__attrs_attrs__", None)
            if (
                attrs_fields is not None
                and not isinstance(message, type)
                and hasattr(message, "__attrs_attrs__")
                and hasattr(message, "__class__")
            ):
                result: dict[str, object] = {}
                for attr_field in attrs_fields:
                    field_name = attr_field.name
                    if hasattr(message, field_name):
                        result[field_name] = getattr(message, field_name)
                return result
            return None

        @classmethod
        def _try_common_serialization_methods(
            cls, message: object
        ) -> dict[str, object] | None:
            """Try common serialization methods."""
            for method_name in ("model_dump", "dict", "as_dict"):
                method = getattr(message, method_name, None)
                if callable(method):
                    try:
                        result_data = method()
                        if FlextMixins.is_dict_like(result_data):
                            return result_data  # type: ignore[return-value]
                    except Exception as e:
                        FlextHandlers._internal_logger.debug(
                            f"Serialization method {method_name} failed: {type(e).__name__}"
                        )
                        continue
            return None

        @classmethod
        def _try_slots_serialization(cls, message: object) -> dict[str, object] | None:
            """Try to serialize object with __slots__."""
            slots = getattr(message, "__slots__", None)
            if not slots:
                return None

            if isinstance(slots, str):
                slot_names: tuple[str, ...] = (slots,)
            elif isinstance(slots, (list, tuple)):
                slot_names = tuple(cast("list[str] | tuple[str, ...]", slots))
            else:
                msg = f"Invalid __slots__ type: {type(slots).__name__}"
                raise FlextExceptions.TypeError(
                    msg,
                    expected_type="str, list, or tuple",
                    actual_type=type(slots).__name__,
                    context=f"message_type: {type(message).__name__}, __slots__: {slots!r}",
                    correlation_id=f"message_serialization_{int(time.time() * 1000)}",
                )

            def get_slot_value(slot_name: str) -> object:
                return getattr(message, slot_name)

            return {
                slot_name: get_slot_value(slot_name)
                for slot_name in slot_names
                if hasattr(message, slot_name)
            }

        @classmethod
        def _raise_invalid_message_type(
            cls, operation: str, context: str, actual_type: str
        ) -> None:
            """Raise TypeError for invalid message type."""
            msg = f"Invalid message type for {operation}: {actual_type}"
            raise FlextExceptions.TypeError(
                msg,
                expected_type=cls._SERIALIZABLE_MESSAGE_EXPECTATION,
                actual_type=actual_type,
                context=f"operation: {context}, message_type: {actual_type}, validation_type: serializable_check",
                correlation_id=f"message_serialization_{int(time.time() * 1000)}",
            )

    @override
    def __init__(self, *, config: FlextModels.Cqrs.Handler) -> None:
        """Initialize handler with simplified single-config approach.

        Args:
            config: Handler configuration object (required)

        """
        super().__init__()

        # Initialize service infrastructure (DI, Context, Logging, Metrics)
        self._init_service(f"flext_handler_{config.handler_name}")

        # Enrich context with handler metadata (Phase 1 enhancement)
        # This automatically adds handler information to all logs
        self._enrich_context(
            handler_name=config.handler_name,
            handler_type=config.handler_type,
            handler_class=self.__class__.__name__,
        )

        self._config_model: FlextModels.Cqrs.Handler = config
        handler_mode_value = (
            config.handler_mode.value
            if hasattr(config.handler_mode, "value")
            else str(config.handler_mode)
        )
        self._execution_context = FlextModels.HandlerExecutionContext(
            handler_name=config.handler_name,
            handler_mode=cast(
                "Literal['command', 'query', 'event', 'operation', 'saga']",
                handler_mode_value,
            ),
        )
        self._accepted_message_types = (
            FlextUtilities.TypeChecker.compute_accepted_message_types(type(self))
        )
        self._revalidate_pydantic_messages = self._extract_revalidation_setting()
        self._type_warning_emitted = False

    def _extract_revalidation_setting(self) -> bool:
        """Extract revalidation setting from configuration."""
        # Check metadata first
        if hasattr(self._config_model, "metadata") and self._config_model.metadata:
            metadata_value = self._config_model.metadata.get(
                "revalidate_pydantic_messages"
            )
            if metadata_value is not None:
                # Handle string values
                if isinstance(metadata_value, str):
                    return metadata_value.lower() in {"true", "1", "yes"}
                return bool(metadata_value)

        # Default to False if not specified
        return False

    def can_handle(self, message_type: object) -> bool:
        """Check if this handler can handle the given message type.

        Implements FlextProtocols.Handler.can_handle() with type checking.

        Args:
            message_type: The type of message to check

        Returns:
            bool: True if this handler can handle the message type

        """
        # Cast to type for internal checking
        if not isinstance(message_type, type):
            return False
        return FlextUtilities.TypeChecker.can_handle_message_type(
            self._accepted_message_types, message_type
        )

    # NOTE: logger property inherited from FlextMixins
    # Provides per-class logger instance via lazy initialization

    @property
    def handler_id(self) -> str:
        """Get handler ID from config.

        Returns:
            str: Handler ID

        """
        return self._config_model.handler_id

    @property
    def handler_name(self) -> str:
        """Get handler name from config.

        Returns:
            str: Handler name

        """
        return self._config_model.handler_name

    @property
    def mode(self) -> str:
        """Get handler mode from config.

        Returns:
            str: Handler mode (command or query)

        """
        # handler_mode is always defined as FlextConstants.HandlerMode.TypeSimple
        return self._config_model.handler_mode

    @property
    def handler_config(self) -> FlextModels.Cqrs.Handler:
        """Get handler configuration.

        Returns:
            FlextModels.Cqrs.Handler: Handler configuration

        """
        return self._config_model

    def validate_command(
        self, command: FlextTypes.AcceptableMessageType
    ) -> FlextResult[None]:
        """Validate a command message with type-safe parameter.

        Args:
            command: The command to validate (typed parameter)

        Returns:
            FlextResult indicating validation success or failure

        """
        return FlextHandlers._MessageValidator.validate_message(
            command,
            operation=FlextConstants.Cqrs.COMMAND_HANDLER_TYPE,
            revalidate_pydantic_messages=self._revalidate_pydantic_messages,
        )

    def validate_query(
        self, query: FlextTypes.AcceptableMessageType
    ) -> FlextResult[None]:
        """Validate a query message with type-safe parameter.

        Args:
            query: The query to validate (typed parameter)

        Returns:
            FlextResult indicating validation success or failure

        """
        return FlextHandlers._MessageValidator.validate_message(
            query,
            operation=FlextConstants.Cqrs.QUERY_HANDLER_TYPE,
            revalidate_pydantic_messages=self._revalidate_pydantic_messages,
        )

    def validate(self, _data: MessageT_contra) -> FlextResult[None]:
        """Validate input data based on handler mode with generic type support.

        Generic validation that delegates to mode-specific validation methods.
        Part of FlextProtocols.Handler protocol implementation.
        Type parameter MessageT_contra ensures type consistency.

        Args:
            _data: The data to validate (generic type MessageT_contra)

        Returns:
            FlextResult[None]: Success if valid, failure with error details

        Examples:
            >>> handler = MyCommandHandler()
            >>> result = handler.validate(command_data)
            >>> if result.is_success:
            ...     # Validation passed
            ...     pass

        """
        if self.mode == FlextConstants.Cqrs.COMMAND_HANDLER_TYPE:
            return self.validate_command(_data)
        if self.mode == FlextConstants.Cqrs.QUERY_HANDLER_TYPE:
            return self.validate_query(_data)
        # For event and saga handlers, use generic validation
        return FlextHandlers._MessageValidator.validate_message(
            cast("object", _data),
            operation=self.mode,
            revalidate_pydantic_messages=self._revalidate_pydantic_messages,
        )

    def execute(self, message: MessageT_contra) -> FlextResult[ResultT]:
        """Execute the handler with the message.

        Args:
            message: The message to execute

        Returns:
            FlextResult containing the execution result or error

        """
        return self._run_pipeline(message, operation=self.mode)

    def __call__(self, input_data: MessageT_contra) -> FlextResult[ResultT]:
        """Callable interface for Application.Handler protocol.

        Makes the handler callable as a function, delegating to execute() method
        for consistent handler invocation. Part of FlextProtocols.Handler
        protocol implementation.

        Args:
            input_data: The input message to process

        Returns:
            FlextResult[ResultT]: Execution result

        Examples:
            >>> handler = MyCommandHandler()
            >>> result = handler(command)  # Callable interface
            >>> # Equivalent to: handler.execute(command)

        """
        return self.execute(input_data)

    def _validate_handler_result(
        self,
        message: MessageT_contra,
        result: FlextResult[ResultT],
    ) -> FlextResult[ResultT]:
        """Validate handler result matches expected types.

        Performs runtime type validation for both message input and result output:
        1. Validates message type matches _expected_message_type (if set)
        2. Validates result value type matches _expected_result_type (if set and successful)

        Args:
            message: The message that was handled
            result: The FlextResult returned by handle()

        Returns:
            FlextResult[ResultT]: Original result if validation passes, failure otherwise

        Examples:
            >>> # ✅ Validation passes for correct types
            >>> message = CreateUserCommand(name="Alice")
            >>> result = self.ok(User(name="Alice"))
            >>> validated = self._validate_handler_result(message, result)
            >>> assert validated.is_success
            >>>
            >>> # ❌ Validation fails for wrong result type
            >>> result = self.ok(Product(id="123"))  # Wrong type!
            >>> validated = self._validate_handler_result(message, result)
            >>> assert validated.is_failure

        """
        # Validate message type
        if self._expected_message_type is not None and not is_bearable(
            message, self._expected_message_type
        ):
            expected_name = getattr(
                self._expected_message_type,
                "__name__",
                str(self._expected_message_type),
            )
            actual_name = type(message).__name__
            msg = (
                f"{self.__class__.__name__}.handle() received message of type "
                f"{actual_name} instead of {expected_name}. "
                f"Message: {message!r}"
            )
            return self.fail(msg, error_code="TYPE_MISMATCH")

        # Validate result type (only on success - errors don't need type validation)
        if (
            self._expected_result_type is not None
            and result.is_success
            and not is_bearable(result.value, self._expected_result_type)
        ):
            expected_name = getattr(
                self._expected_result_type,
                "__name__",
                str(self._expected_result_type),
            )
            actual_name = type(result.value).__name__
            msg = (
                f"{self.__class__.__name__}.handle() returned "
                f"FlextResult[{actual_name}] instead of "
                f"FlextResult[{expected_name}]. "
                f"Data: {result.value!r}"
            )
            return self.fail(msg, error_code="TYPE_MISMATCH")

        return result

    def _run_pipeline(
        self,
        message: MessageT_contra | dict[str, object],
        operation: str = "command",
    ) -> FlextResult[ResultT]:
        """Run the handler pipeline with message processing.

        Args:
            message: The message to process
            operation: The operation type (command or query)

        Returns:
            FlextResult containing the processing result or error

        """
        # Extract message ID
        if self.is_dict_like(message):
            (
                str(message.get(f"{operation}_id", "unknown"))
                or str(message.get("message_id", "unknown"))
                or "unknown"
            )
        elif hasattr(message, f"{operation}_id"):
            str(getattr(message, f"{operation}_id", "unknown"))
        elif hasattr(message, "message_id"):
            str(getattr(message, "message_id", "unknown"))

        message_type: str = str(type(message).__name__)

        # Validate operation matches handler mode
        if operation != self.mode:
            return self.fail(
                f"Handler with mode '{self.mode}' cannot execute {operation} pipelines"
            )

        # Validate message can be handled
        message_type_obj: type[object] = type(message)
        if not self.can_handle(cast("type[MessageT_contra]", message_type_obj)):
            return self.fail(f"Handler cannot handle message type {message_type}")

        # Validate message
        validation_result = (
            self.validate_command(cast("object", message))
            if operation == "command"
            else self.validate_query(cast("object", message))
        )
        if validation_result.is_failure:
            return self.fail(f"Message validation failed: {validation_result.error}")

        # Execute handler with runtime type validation
        try:
            result = self.handle(cast("MessageT_contra", message))
            # Validate result types if __class_getitem__ was used
            return self._validate_handler_result(
                cast("MessageT_contra", message), result
            )
        except Exception as e:
            # VALIDATION HIERARCHY - User handler execution (CRITICAL)
            # User-registered handlers can raise any exception
            # Safe behavior: capture error, return failure result
            return self.fail(f"Critical handler failure: {e!s}")

    @abstractmethod
    def handle(self, message: MessageT_contra) -> FlextResult[ResultT]:
        """Handle a message and return a result.

        Args:
            message: The message to handle

        Returns:
            FlextResult containing the result or error

        """
        ...

    @classmethod
    def from_callable(  # type: ignore[override]
        cls,
        func: FlextTypes.HandlerCallableType,
        handler_name: str | None = None,
        handler_type: FlextConstants.Cqrs.HandlerType = FlextConstants.Cqrs.COMMAND_HANDLER_TYPE,
        mode: str | None = None,
        handler_config: FlextModels.Cqrs.Handler | None = None,
    ) -> FlextHandlers[object, object]:
        """Create a handler from a callable function.

        Args:
            func: The callable function to wrap
            handler_name: Name for the handler (defaults to function name)
            handler_type: Type of handler (command, query, etc.)
            mode: Handler mode (for compatibility)
            handler_config: Optional handler configuration model (must be FlextModels.Cqrs.Handler)

        Returns:
            A FlextHandlers instance wrapping the callable

        """
        # Ensure handler_name is always a string
        resolved_handler_name: str = (
            handler_name
            if handler_name is not None
            else getattr(func, "__name__", "unknown_handler")
        )

        # Use mode if provided (compatibility), otherwise use handler_type
        effective_type: FlextConstants.Cqrs.HandlerType = (
            cast("FlextConstants.Cqrs.HandlerType", mode)
            if mode is not None
            else handler_type
        )

        # Validate mode/handler_type
        if effective_type not in {
            FlextConstants.Cqrs.COMMAND_HANDLER_TYPE,
            FlextConstants.Cqrs.QUERY_HANDLER_TYPE,
        }:
            msg = f"Invalid handler mode: {effective_type}. Must be '{FlextConstants.Cqrs.COMMAND_HANDLER_TYPE}' or '{FlextConstants.Cqrs.QUERY_HANDLER_TYPE}'"
            raise FlextExceptions.ValidationError(
                message=msg,
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        # Use provided config or create default
        if handler_config is not None:
            config = handler_config
        else:
            try:
                config = FlextModels.Cqrs.Handler(
                    handler_id=f"{resolved_handler_name}_{id(func)}",
                    handler_name=resolved_handler_name,
                    handler_type=effective_type,
                    handler_mode=effective_type,
                )
            except Exception as e:
                # VALIDATION HIERARCHY - Handler config creation (MEDIUM)
                # Model creation can raise validation exceptions
                # Safe behavior: re-raise as ValidationError with context
                msg = f"Failed to create handler config: {e}"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                ) from e

        # Create a wrapper class with proper generic types
        class CallableHandler(FlextHandlers[CallableInputT, CallableOutputT]):
            def __init__(self, config: FlextModels.Cqrs.Handler) -> None:
                super().__init__(config=config)
                # Store callable as object - type variables are method-local, not class-scoped
                self.original_callable: object = func

            @override
            def handle(self, message: CallableInputT) -> FlextResult[CallableOutputT]:
                try:
                    result = func(message)
                    if isinstance(result, FlextResult):
                        return cast("FlextResult[CallableOutputT]", result)
                    return self.ok(cast("CallableOutputT", result))
                except Exception as e:
                    # VALIDATION HIERARCHY - User callable execution (CRITICAL)
                    # User-provided callable can raise any exception
                    # Safe behavior: wrap exception in self.fail()
                    return self.fail(str(e))

            @override
            def can_handle(self, message_type: object) -> bool:
                """Override can_handle for callable wrappers to enable auto-discovery.

                Callable wrappers created via from_callable may have method-local TypeVars
                that can't be introspected by compute_accepted_message_types(), resulting
                in empty _accepted_message_types. Instead, we rely on validation to catch
                type incompatibility, allowing auto-discovery to find this handler.
                """
                # Always return True for auto-discovery - validation will fail if incompatible
                return True

        return CallableHandler(config=config)

    # =========================================================================
    # Protocol Implementation: MessageValidator, MetricsCollector, ExecutionContextManager
    # =========================================================================

    def validate_message(self, message: MessageT_contra) -> FlextResult[None]:
        """Validate message (MessageValidator protocol).

        Part of MessageValidator protocol implementation.

        Args:
            message: Message to validate

        Returns:
            FlextResult[None]: Validation success or error

        """
        return self.validate(message)

    def record_metric(self, name: str, value: float) -> FlextResult[None]:
        """Record metric (MetricsCollector protocol).

        Part of MetricsCollector protocol implementation.

        Args:
            name: Metric name
            value: Metric value

        Returns:
            FlextResult[None]: Success or error

        """
        try:
            if not hasattr(self, "_metrics"):
                self._metrics: dict[str, float] = {}
            self._metrics[name] = value
            return self.ok(None)
        except Exception as e:
            # VALIDATION HIERARCHY - Metrics recording (MEDIUM)
            # Dict operations with user data can raise any exception
            # Safe behavior: wrap in FlextResult with error code
            return self.fail(
                f"Metric recording failed: {e}",
                error_code="METRICS_ERROR",
            )

    def get_metrics(self) -> FlextResult[dict[str, object]]:
        """Get metrics (MetricsCollector protocol).

        Part of MetricsCollector protocol implementation.

        Returns:
            FlextResult[dict]: Metrics dictionary or error

        """
        try:
            metrics = getattr(self, "_metrics", {})
            return FlextResult[dict[str, object]].ok(metrics)
        except Exception as e:
            # VALIDATION HIERARCHY - Metrics retrieval (MEDIUM)
            # Getattr operations can raise unexpected exceptions
            # Safe behavior: wrap in FlextResult with error code
            return FlextResult[dict[str, object]].fail(
                f"Metrics retrieval failed: {e}",
                error_code="METRICS_ERROR",
            )

    def push_context(self, context: dict[str, object]) -> FlextResult[None]:
        """Push context (ExecutionContextManager protocol).

        Part of ExecutionContextManager protocol implementation.

        Args:
            context: Context dictionary

        Returns:
            FlextResult[None]: Success or error

        """
        try:
            if not hasattr(self, "_context_stack"):
                self._context_stack: list[dict[str, object]] = []
            self._context_stack.append(context)
            return self.ok(None)
        except Exception as e:
            # VALIDATION HIERARCHY - Context push (MEDIUM)
            # List operations with user data can raise any exception
            # Safe behavior: wrap in FlextResult with error code
            return self.fail(
                f"Context push failed: {e}",
                error_code="CONTEXT_ERROR",
            )

    def pop_context(self) -> FlextResult[None]:
        """Pop context (ExecutionContextManager protocol).

        Part of ExecutionContextManager protocol implementation.

        Returns:
            FlextResult[None]: Success or error

        """
        try:
            if not hasattr(self, "_context_stack"):
                self._context_stack = []
            if self._context_stack:
                self._context_stack.pop()
            return self.ok(None)
        except Exception as e:
            # VALIDATION HIERARCHY - Context pop (MEDIUM)
            # List operations can raise unexpected exceptions
            # Safe behavior: wrap in FlextResult with error code
            return self.fail(
                f"Context pop failed: {e}",
                error_code="CONTEXT_ERROR",
            )


__all__: list[str] = [
    "FlextHandlers",
]
