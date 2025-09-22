"""Unified CQRS handler base promoted for the FLEXT 1.0.0 rollout.

Handlers expose the validation and telemetry hooks referenced in
``docs/architecture.md`` so downstream packages can migrate to
``FlextDispatcher`` without bespoke wiring.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from abc import abstractmethod
from typing import Any, Literal, TypeVar, get_origin, get_type_hints

from pydantic import BaseModel

from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextHandlers[MessageT, ResultT](FlextMixins):
    """Generic base that normalises command/query handler behaviour.

    The class codifies how handlers are registered, validated, and measured so
    that ``FlextBus`` and ``FlextDispatcher`` see a consistent surface across
    the ecosystem, fulfilling the unified dispatcher pillar.

    Implements FlextProtocols.Commands.CommandHandler and QueryHandler protocols
    for proper CQRS pattern compliance with enhanced type safety.
    """

    # Note: Use FlextConstants.Cqrs.DEFAULT_HANDLER_TYPE directly instead of wrapper

    def __init__(
        self,
        *,
        config: FlextModels.CqrsConfig.Handler | None = None,
        # Legacy parameter support for backward compatibility (will be deprecated)
        handler_mode: Literal["command", "query"] | None = None,
        handler_name: str | None = None,
        handler_id: str | None = None,
        handler_config: FlextModels.CqrsConfig.Handler
        | dict[str, object]
        | None = None,
        command_timeout: int = 0,
        max_command_retries: int = 0,
    ) -> None:
        """Initialize handler with consolidated Pydantic validation.

        Args:
            config: Complete handler configuration model (preferred approach)
            handler_mode: DEPRECATED - Handler type (command/query)
            handler_name: DEPRECATED - Handler name
            handler_id: DEPRECATED - Handler ID
            handler_config: DEPRECATED - Handler configuration
            command_timeout: DEPRECATED - Command timeout
            max_command_retries: DEPRECATED - Max retries

        """
        super().__init__()

        # Use the new consolidated config approach if provided
        if config is not None:
            self._config_model = config
        else:
            # Handle legacy parameters by creating config model
            resolved_name = handler_name or self.__class__.__name__
            resolved_mode = self._resolve_mode(handler_mode, handler_config)

            # Convert handler_config to dict if it's a Handler object
            config_dict: dict[str, object] | None = None
            if handler_config is not None:
                if isinstance(handler_config, dict):
                    config_dict = handler_config
                elif hasattr(handler_config, "model_dump"):
                    # Type-safe call to model_dump for Pydantic models
                    try:
                        dump_result = handler_config.model_dump()
                        config_dict = (
                            dump_result  # model_dump() always returns dict[str, object]
                        )
                    except Exception:
                        config_dict = {}

            self._config_model = FlextModels.CqrsConfig.Handler.create_handler_config(
                handler_type=resolved_mode,
                default_name=resolved_name,
                default_id=handler_id,
                handler_config=config_dict,
                command_timeout=command_timeout,
                max_command_retries=max_command_retries,
            )

        # Initialize internal state from config model (removed redundant _config)
        self._handler_name = self._config_model.handler_name
        self.handler_id = self._config_model.handler_id
        self._start_time: float | None = None
        self._metrics_state: FlextTypes.Core.Dict | None = None
        self._accepted_message_types = self._resolve_message_types()
        self._type_warning_emitted = False

    def _resolve_mode(
        self,
        handler_mode: Literal["command", "query"] | None,
        handler_config: FlextModels.CqrsConfig.Handler | dict[str, object] | None,
    ) -> Literal["command", "query"]:
        """Resolve handler mode using FlextConstants defaults.

        Returns:
            Literal mode type ('command' or 'query') resolved from parameters or defaults.

        """
        if handler_mode in {
            FlextConstants.Cqrs.COMMAND_HANDLER_TYPE,
            FlextConstants.Cqrs.QUERY_HANDLER_TYPE,
        }:
            return handler_mode
        if isinstance(handler_config, FlextModels.CqrsConfig.Handler):
            return handler_config.handler_type
        if isinstance(handler_config, dict):
            raw_mode = handler_config.get("handler_type")
            if raw_mode == FlextConstants.Cqrs.COMMAND_HANDLER_TYPE:
                return FlextConstants.Cqrs.COMMAND_HANDLER_TYPE
            if raw_mode == FlextConstants.Cqrs.QUERY_HANDLER_TYPE:
                return FlextConstants.Cqrs.QUERY_HANDLER_TYPE
        return FlextConstants.Cqrs.DEFAULT_HANDLER_TYPE

    @property
    def mode(self) -> Literal["command", "query"]:
        """Return configured handler mode from the config model."""
        return self._config_model.handler_type

    @property
    def handler_name(self) -> str:
        """Get handler name for identification."""
        return str(self._handler_name)

    @property
    def logger(self) -> FlextLogger:
        """Get logger instance for this handler."""
        return FlextLogger(self.__class__.__name__)

    @property
    def config(self) -> FlextModels.CqrsConfig.Handler:
        """Return validated handler configuration model."""
        return self._config_model

    def can_handle(self, message_type: object) -> bool:
        """Check if handler can process this message type.

        Implements FlextProtocols.Commands.CommandHandler.can_handle
        and FlextProtocols.Commands.QueryHandler equivalent for type checking.

        Returns:
            True if handler can process the message type, False otherwise.

        """
        self.logger.debug(
            "checking_handler_capability",
            handler_mode=self.mode,
            message_type_name=getattr(message_type, "__name__", str(message_type)),
        )

        if not self._accepted_message_types:
            if not self._type_warning_emitted:
                self.logger.warning(
                    "handler_type_constraints_unknown",
                    handler_name=self.handler_name,
                    handler_class=self.__class__.__name__,
                )
                self._type_warning_emitted = True
            return False

        for expected_type in self._accepted_message_types:
            can_handle_result = self._evaluate_type_compatibility(
                expected_type, message_type
            )

            self.logger.debug(
                "handler_type_check",
                can_handle=can_handle_result,
                expected_type=getattr(expected_type, "__name__", str(expected_type)),
            )

            if can_handle_result:
                return True

        return False

    def _resolve_message_types(self) -> tuple[object, ...]:
        """Determine the accepted message types for this handler instance."""

        message_types: list[object] = []

        for cls in self.__class__.__mro__:
            orig_bases = getattr(cls, "__orig_bases__", ())
            for base in orig_bases or ():
                message_types.extend(self._extract_message_types_from_base(base))

        if not message_types:
            message_types.extend(self._resolve_message_types_from_hints())

        return self._normalise_message_types(message_types)

    def _extract_message_types_from_base(self, base: object) -> list[object]:
        """Extract message type arguments from a generic base definition."""

        args = getattr(base, "__args__", None)
        if not args:
            return []

        origin = get_origin(base) or getattr(base, "__origin__", None) or base

        try:
            if isinstance(origin, type) and issubclass(origin, FlextHandlers):
                return [args[0]]
        except TypeError:
            return []

        return []

    def _resolve_message_types_from_hints(self) -> list[object]:
        """Resolve message types from explicit type hints when generics are absent."""

        hint_sources = (
            ("handle", "message"),
            ("handle_command", "command"),
            ("handle_query", "query"),
        )

        resolved: list[object] = []

        for method_name, parameter_name in hint_sources:
            method = getattr(self.__class__, method_name, None)
            if method is None:
                continue

            try:
                type_hints = get_type_hints(method, include_extras=True)
            except Exception:
                continue

            hint = type_hints.get(parameter_name)
            if hint is not None:
                resolved.append(hint)

        return resolved

    def _normalise_message_types(self, message_types: list[object]) -> tuple[object, ...]:
        """Deduplicate and filter inferred message types."""

        unique: list[object] = []
        for message_type in message_types:
            if message_type is None:
                continue
            if message_type is Any:
                continue
            if message_type is object:
                continue
            if isinstance(message_type, TypeVar):
                continue
            if message_type in unique:
                continue
            unique.append(message_type)

        return tuple(unique)

    def _evaluate_type_compatibility(
        self, expected_type: object, message_type: object
    ) -> bool:
        """Evaluate compatibility between expected and actual message types.

        Returns:
            bool: True if types are compatible, False otherwise.

        """
        origin_type = get_origin(expected_type) or expected_type
        message_origin = get_origin(message_type) or message_type

        if isinstance(message_type, type) or hasattr(message_type, "__origin__"):
            return self._handle_type_or_origin_check(
                expected_type, message_type, origin_type, message_origin
            )
        return self._handle_instance_check(message_type, origin_type)

    def _handle_type_or_origin_check(
        self,
        expected_type: object,
        message_type: object,
        origin_type: object,
        message_origin: object,
    ) -> bool:
        """Handle type checking for types or objects with __origin__.

        Returns:
            bool: True if type check passes, False otherwise.

        """
        try:
            if hasattr(message_type, "__origin__"):
                return message_origin == origin_type
            if isinstance(message_type, type) and isinstance(origin_type, type):
                return issubclass(message_type, origin_type)
            return message_type == expected_type
        except TypeError:
            return message_type == expected_type

    def _handle_instance_check(self, message_type: object, origin_type: object) -> bool:
        """Handle instance checking for non-type objects.

        Returns:
            bool: True if instance check passes or TypeError occurs, False otherwise.

        """
        try:
            if isinstance(origin_type, type):
                return isinstance(message_type, origin_type)
            return True
        except TypeError:
            return True

    def validate_command(self, command: object) -> FlextResult[None]:
        """Validate command using enhanced Pydantic 2 validation and FlextExceptions.

        Returns:
            FlextResult[None] indicating success or failure with error details.

        """
        return self._validate_message(
            command,
            operation=FlextConstants.Cqrs.COMMAND_HANDLER_TYPE,
        )

    def validate_query(self, query: object) -> FlextResult[None]:
        """Validate query using enhanced Pydantic 2 validation and FlextExceptions.

        Returns:
            FlextResult[None] indicating success or failure with error details.

        """
        return self._validate_message(
            query,
            operation=FlextConstants.Cqrs.QUERY_HANDLER_TYPE,
        )

    def _validate_message(
        self,
        message: object,
        *,
        operation: Literal["command", "query"],
    ) -> FlextResult[None]:
        """Validate message using Pydantic model validation, FlextUtilities, and FlextExceptions.

        Returns:
            FlextResult[None] indicating validation success or failure with error details.

        """
        # First check if the message has built-in validation methods
        method_name = f"validate_{operation}"
        validate_method = getattr(message, method_name, None)

        if callable(validate_method):
            try:
                result = validate_method()
                # Handle FlextResult-like objects
                if hasattr(result, "is_success") and hasattr(result, "is_failure"):
                    if getattr(result, "is_failure", False):
                        error_msg = getattr(
                            result, "error", f"{operation.title()} validation failed"
                        )
                        error_code = (
                            getattr(result, "error_code", None)
                            or FlextConstants.Errors.VALIDATION_ERROR
                        )
                        error_data = getattr(result, "error_data", {})
                        return FlextResult[None].fail(
                            error_msg,
                            error_code=error_code,
                            error_data=error_data,
                        )
                    return FlextResult[None].ok(None)
                # Handle boolean results
                if isinstance(result, bool):
                    return (
                        FlextResult[None].ok(None)
                        if result
                        else FlextResult[None].fail(
                            f"{operation.title()} validation failed",
                            error_code=FlextConstants.Errors.VALIDATION_ERROR,
                        )
                    )
            except Exception as e:
                # Use FlextExceptions.ValidationError for structured error handling
                validation_error = FlextExceptions.ValidationError(
                    f"{operation.title()} validation method failed: {e}",
                    field=method_name,
                    value=str(message)[:100]
                    if hasattr(message, "__str__")
                    else "unknown",
                    validation_details={
                        "original_exception": str(e),
                        "method_name": method_name,
                    },
                    context={
                        "operation": operation,
                        "message_type": type(message).__name__,
                        "validation_method": method_name,
                    },
                    correlation_id=f"validation_{int(time.time() * 1000)}",
                )

                return FlextResult[None].fail(
                    str(validation_error),
                    error_code=validation_error.error_code,
                    error_data={"exception_context": validation_error.context},
                )

        # If message is a Pydantic model, it's already validated
        if isinstance(message, BaseModel):
            try:
                # Re-validate to ensure consistency
                message_data = message.model_dump()
                # Use the class's model_validate method
                message.__class__.model_validate(message_data)
                return FlextResult[None].ok(None)
            except Exception as e:
                # Use FlextExceptions.ValidationError for Pydantic validation failures
                pydantic_error = FlextExceptions.ValidationError(
                    f"Pydantic validation failed: {e}",
                    field="pydantic_model",
                    value=str(message)[:100]
                    if hasattr(message, "__str__")
                    else "unknown",
                    validation_details={
                        "pydantic_exception": str(e),
                        "model_class": message.__class__.__name__,
                    },
                    context={
                        "operation": operation,
                        "message_type": type(message).__name__,
                        "validation_type": "pydantic",
                    },
                    correlation_id=f"pydantic_validation_{int(time.time() * 1000)}",
                )

                return FlextResult[None].fail(
                    str(pydantic_error),
                    error_code=pydantic_error.error_code,
                    error_data={"exception_context": pydantic_error.context},
                )

        # For non-Pydantic objects, basic validation using FlextExceptions
        if not hasattr(message, "__dict__") and not isinstance(
            message, (dict, str, int, float, bool)
        ):
            type_error = FlextExceptions.TypeError(
                f"Invalid message type for {operation}: {type(message).__name__}",
                expected_type="dict, str, int, float, bool, or object with __dict__",
                actual_type=type(message).__name__,
                context={
                    "operation": operation,
                    "message_type": type(message).__name__,
                    "validation_type": "basic_type_check",
                },
                correlation_id=f"type_validation_{int(time.time() * 1000)}",
            )

            return FlextResult[None].fail(
                str(type_error),
                error_code=type_error.error_code,
                error_data={"exception_context": type_error.context},
            )

        return FlextResult[None].ok(None)

    @abstractmethod
    def handle(self, message: MessageT) -> FlextResult[ResultT]:
        """Handle the message and return result.

        Subclasses must override this method.

        Implements FlextProtocols.Commands.CommandHandler.handle
        and FlextProtocols.Commands.QueryHandler.handle protocols.
        """
        ...

    def execute(self, message: MessageT) -> FlextResult[ResultT]:
        """Execute message (command or query) with full validation and error handling.

        This unified execution method automatically determines the operation type based on
        the handler's configured mode and executes the appropriate pipeline.

        Args:
            message: The command or query to execute.

        Returns:
            A FlextResult containing the execution result or error details.

        """
        return self._run_pipeline(message, operation=self.mode)

    def handle_query(self, query: MessageT) -> FlextResult[ResultT]:
        """Execute query with validation and error handling.

        This is a semantic alias for execute() when working with query handlers.

        Args:
            query: The query to execute.

        Returns:
            A FlextResult containing the query result or error details.

        """
        return self.execute(query)

    def handle_command(self, command: MessageT) -> FlextResult[ResultT]:
        """Execute command with validation and error handling.

        This is a semantic alias for execute() when working with command handlers.

        Args:
            command: The command to execute.

        Returns:
            A FlextResult containing the execution result or error details.

        """
        return self.execute(command)

    def _run_pipeline(
        self,
        message: MessageT,
        *,
        operation: Literal["command", "query"],
    ) -> FlextResult[ResultT]:
        """Execute handler pipeline using railway-oriented programming.

        Returns:
            FlextResult[ResultT] containing the execution result or error details.

        """
        message_type = type(message).__name__
        identifier = getattr(
            message,
            FlextConstants.Cqrs.COMMAND_HANDLER_TYPE + "_id"
            if operation == FlextConstants.Cqrs.COMMAND_HANDLER_TYPE
            else FlextConstants.Cqrs.QUERY_HANDLER_TYPE + "_id",
            getattr(message, "id", FlextConstants.Messages.UNKNOWN_ERROR),
        )

        # Log start of pipeline
        self.logger.info(
            "starting_handler_pipeline",
            handler_mode=self.mode,
            message_type=message_type,
            message_id=identifier,
        )

        # Use railway pattern for validation chain
        validation_result = FlextResult.chain_validations(
            lambda: self._validate_mode(operation),
            lambda: self._validate_can_handle(message),
            lambda: self._validate_message(message, operation=operation),
        )

        if validation_result.is_failure:
            return FlextResult[ResultT].fail(
                validation_result.error or "Validation chain failed",
                error_code=validation_result.error_code,
                error_data=validation_result.error_data,
            )

        return self._execute_with_timing(message, message_type, identifier)

    def _validate_mode(
        self, operation: Literal["command", "query"]
    ) -> FlextResult[None]:
        """Validate handler mode matches operation type using FlextConstants.

        Returns:
            FlextResult[None] indicating validation success or failure.

        """
        if self.mode != operation:
            error_msg = (
                f"{self.handler_name} is configured for {self.mode} operations "
                f"and cannot execute {operation} pipelines"
            )
            self.logger.error(
                "invalid_handler_mode",
                error_message=error_msg,
                expected_mode=operation,
                actual_mode=self.mode,
            )
            return FlextResult[None].fail(
                error_msg,
                error_code=FlextConstants.Errors.COMMAND_HANDLER_NOT_FOUND,
            )
        return FlextResult[None].ok(None)

    def _validate_can_handle(self, message: MessageT) -> FlextResult[None]:
        """Validate handler can process this message type using FlextConstants.

        Returns:
            FlextResult[None] indicating validation success or failure.

        """
        if not self.can_handle(type(message)):
            message_type = type(message).__name__
            error_msg = f"{self.handler_name} cannot handle {message_type}"
            self.logger.error(
                "handler_cannot_handle",
                error_message=error_msg,
                handler_name=self.handler_name,
                message_type=message_type,
            )
            return FlextResult[None].fail(
                error_msg,
                error_code=FlextConstants.Errors.COMMAND_HANDLER_NOT_FOUND,
            )
        return FlextResult[None].ok(None)

    def _execute_with_timing(
        self, message: MessageT, message_type: str, identifier: str
    ) -> FlextResult[ResultT]:
        """Execute handler with timing and logging using FlextUtilities and FlextExceptions.

        Returns:
            FlextResult[ResultT] containing the execution result or error details.

        """
        self._start_time = time.time()

        try:
            self.logger.debug(
                "processing_message",
                handler_mode=self.mode,
                message_type=message_type,
                message_id=identifier,
            )

            result: FlextResult[ResultT] = self.handle(message)

            elapsed = time.time() - (self._start_time or 0.0)
            execution_time_ms = round(elapsed * 1000, 2) if elapsed else 0

            self.logger.info(
                "handler_pipeline_completed",
                handler_mode=self.mode,
                message_type=message_type,
                message_id=identifier,
                execution_time_ms=execution_time_ms,
                success=result.is_success,
            )
            return result

        except TypeError as exc:
            # Handle type-related errors with structured FlextExceptions
            elapsed = time.time() - (self._start_time or 0.0)
            execution_time_ms = round(elapsed * 1000, 2) if elapsed else 0

            # Create structured type error with FlextExceptions
            type_error = FlextExceptions.TypeError(
                f"Handler type error during {self.mode} processing: {exc}",
                expected_type=getattr(message, "__class__", {}).get(
                    "__name__", "unknown"
                ),
                actual_type=type(message).__name__,
                context={
                    "handler_mode": self.mode,
                    "message_type": message_type,
                    "message_id": identifier,
                    "execution_time_ms": execution_time_ms,
                },
                correlation_id=f"handler_{identifier}_{int(time.time() * 1000)}",
            )

            self.logger.exception(
                "handler_type_error",
                handler_mode=self.mode,
                message_type=message_type,
                message_id=identifier,
                execution_time_ms=execution_time_ms,
                error_code=type_error.error_code,
                correlation_id=type_error.correlation_id,
            )

            return FlextResult[ResultT].fail(
                str(type_error),
                error_code=type_error.error_code,
                error_data={"exception_context": type_error.context},
            )

        except (ValueError, AttributeError) as exc:
            # Handle validation and attribute errors with structured FlextExceptions
            elapsed = time.time() - (self._start_time or 0.0)
            execution_time_ms = round(elapsed * 1000, 2) if elapsed else 0

            if isinstance(exc, ValueError):
                validation_error = FlextExceptions.ValidationError(
                    f"Handler validation error during {self.mode} processing: {exc}",
                    field=getattr(message, "__class__", {}).get("__name__", "message"),
                    value=str(message)[:100]
                    if hasattr(message, "__str__")
                    else "unknown",
                    validation_details={"original_exception": str(exc)},
                    context={
                        "handler_mode": self.mode,
                        "message_type": message_type,
                        "message_id": identifier,
                        "execution_time_ms": execution_time_ms,
                    },
                    correlation_id=f"handler_{identifier}_{int(time.time() * 1000)}",
                )
                error_code = validation_error.error_code
                error_context = validation_error.context
            else:  # AttributeError
                attribute_error = FlextExceptions.AttributeError(
                    f"Handler attribute error during {self.mode} processing: {exc}",
                    attribute_name=getattr(exc, "name", "unknown"),
                    attribute_context={
                        "handler_mode": self.mode,
                        "message_type": message_type,
                        "message_id": identifier,
                        "execution_time_ms": execution_time_ms,
                    },
                    correlation_id=f"handler_{identifier}_{int(time.time() * 1000)}",
                )
                error_code = attribute_error.error_code
                error_context = attribute_error.context

            self.logger.exception(
                "handler_validation_or_attribute_error",
                handler_mode=self.mode,
                message_type=message_type,
                message_id=identifier,
                execution_time_ms=execution_time_ms,
                error_code=error_code,
            )

            return FlextResult[ResultT].fail(
                f"Handler {exc.__class__.__name__.lower()} during {self.mode} processing: {exc}",
                error_code=error_code,
                error_data={"exception_context": error_context},
            )

        except RuntimeError as exc:
            # Handle runtime errors with structured FlextExceptions
            elapsed = time.time() - (self._start_time or 0.0)
            execution_time_ms = round(elapsed * 1000, 2) if elapsed else 0

            processing_error = FlextExceptions.ProcessingError(
                f"Handler runtime error during {self.mode} processing: {exc}",
                business_rule=f"{self.mode}_processing",
                operation=f"handle_{self.mode}",
                context={
                    "handler_mode": self.mode,
                    "message_type": message_type,
                    "message_id": identifier,
                    "execution_time_ms": execution_time_ms,
                    "handler_name": self.handler_name,
                },
                correlation_id=f"handler_{identifier}_{int(time.time() * 1000)}",
            )

            self.logger.exception(
                "handler_runtime_error",
                handler_mode=self.mode,
                message_type=message_type,
                message_id=identifier,
                execution_time_ms=execution_time_ms,
                error_code=processing_error.error_code,
                correlation_id=processing_error.correlation_id,
            )

            return FlextResult[ResultT].fail(
                str(processing_error),
                error_code=processing_error.error_code,
                error_data={"exception_context": processing_error.context},
            )

        except Exception as exc:
            # Handle any other unexpected exceptions with FlextExceptions.CriticalError
            elapsed = time.time() - (self._start_time or 0.0)
            execution_time_ms = round(elapsed * 1000, 2) if elapsed else 0

            critical_error = FlextExceptions.CriticalError(
                f"Critical handler failure during {self.mode} processing: {exc}",
                context={
                    "handler_mode": self.mode,
                    "message_type": message_type,
                    "message_id": identifier,
                    "execution_time_ms": execution_time_ms,
                    "handler_name": self.handler_name,
                    "exception_type": type(exc).__name__,
                    "original_exception": str(exc),
                },
                correlation_id=f"handler_{identifier}_{int(time.time() * 1000)}",
            )

            self.logger.exception(
                "handler_critical_failure",
                handler_mode=self.mode,
                message_type=message_type,
                message_id=identifier,
                execution_time_ms=execution_time_ms,
                exception_type=type(exc).__name__,
                error_code=critical_error.error_code,
                correlation_id=critical_error.correlation_id,
            )

            return FlextResult[ResultT].fail(
                str(critical_error),
                error_code=critical_error.error_code,
                error_data={"exception_context": critical_error.context},
            )


__all__: FlextTypes.Core.StringList = [
    "FlextHandlers",
]
