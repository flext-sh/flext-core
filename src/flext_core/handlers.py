"""Unified handler base for FLEXT CQRS architecture."""

from __future__ import annotations

import time
from typing import ClassVar, Literal, cast, get_origin

from flext_core.constants import FlextConstants
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextHandlers[MessageT, ResultT](FlextMixins):
    """Single handler class supporting command or query operations."""

    DEFAULT_MODE: ClassVar[Literal["command", "query"]] = "command"

    def __init__(
        self,
        *,
        handler_mode: Literal["command", "query"] | None = None,
        handler_name: str | None = None,
        handler_id: str | None = None,
        handler_config: FlextModels.CqrsConfig.Handler
        | dict[str, object]
        | None = None,
        command_timeout: int = 0,
        max_command_retries: int = 0,
    ) -> None:
        """Initialize handler with centralized Pydantic validation."""
        super().__init__()

        self._handler_mode = self._resolve_mode(handler_mode, handler_config)
        resolved_name = handler_name or self.__class__.__name__

        config_model = FlextModels.CqrsConfig.create_handler_config(
            handler_type=self._handler_mode,
            default_name=resolved_name,
            default_id=handler_id,
            handler_config=handler_config,
            command_timeout=command_timeout,
            max_command_retries=max_command_retries,
        )

        self._config_model = config_model
        self._config = config_model.model_dump()
        self._handler_name = config_model.handler_name
        self.handler_id = config_model.handler_id
        self._start_time: float | None = None
        self._metrics_state: FlextTypes.Core.Dict | None = None

    def _resolve_mode(
        self,
        handler_mode: Literal["command", "query"] | None,
        handler_config: FlextModels.CqrsConfig.Handler | dict[str, object] | None,
    ) -> Literal["command", "query"]:
        if handler_mode in {"command", "query"}:
            return handler_mode
        if isinstance(handler_config, FlextModels.CqrsConfig.Handler):
            return handler_config.handler_type
        if isinstance(handler_config, dict):
            raw_mode = handler_config.get("handler_type")
            if raw_mode in {"command", "query"}:
                return cast("Literal['command', 'query']", raw_mode)
        default_mode = getattr(
            self.__class__, "DEFAULT_MODE", FlextHandlers.DEFAULT_MODE
        )
        return cast("Literal['command', 'query']", default_mode)

    @property
    def mode(self) -> Literal["command", "query"]:
        """Return configured handler mode."""
        return self._config_model.handler_type

    @property
    def handler_name(self) -> str:
        """Get handler name for identification."""
        return (
            str(self._handler_name)
            if self._handler_name is not None
            else self.__class__.__name__
        )

    @property
    def logger(self) -> FlextLogger:
        """Get logger instance for this handler."""
        return FlextLogger(self.__class__.__name__)

    @property
    def config(self) -> FlextModels.CqrsConfig.Handler:
        """Return validated handler configuration."""
        return self._config_model

    def can_handle(self, message_type: object) -> bool:
        """Check if handler can process this message type."""
        self.logger.debug(
            "checking_handler_capability",
            handler_mode=self.mode,
            message_type_name=getattr(message_type, "__name__", str(message_type)),
        )

        orig_bases = getattr(self, "__orig_bases__", None)
        if orig_bases is not None:
            for base in orig_bases:
                args = getattr(base, "__args__", None)
                if args is not None and len(args) >= 1:
                    expected_type = base.__args__[0]

                    origin_type = get_origin(expected_type) or expected_type
                    message_origin = get_origin(message_type) or message_type

                    if isinstance(message_type, type) or hasattr(
                        message_type, "__origin__"
                    ):
                        try:
                            if hasattr(message_type, "__origin__"):
                                can_handle_result = message_origin == origin_type
                            elif isinstance(message_type, type) and isinstance(
                                origin_type, type
                            ):
                                can_handle_result = issubclass(
                                    message_type, origin_type
                                )
                            else:
                                can_handle_result = message_type == expected_type
                        except TypeError:
                            can_handle_result = message_type == expected_type
                    else:
                        try:
                            can_handle_result = isinstance(message_type, origin_type)
                        except TypeError:
                            can_handle_result = True

                    self.logger.debug(
                        "handler_type_check",
                        can_handle=can_handle_result,
                        expected_type=getattr(
                            expected_type, "__name__", str(expected_type)
                        ),
                    )
                    return bool(can_handle_result)

        self.logger.info("handler_type_constraints_unknown")
        return True

    def validate_command(self, command: object) -> FlextResult[None]:
        """Validate command prior to handling."""
        return self._validate_message(command, operation="command")

    def validate_query(self, query: object) -> FlextResult[None]:
        """Validate query prior to handling."""
        return self._validate_message(query, operation="query")

    def _validate_message(
        self,
        message: object,
        *,
        operation: Literal["command", "query"],
    ) -> FlextResult[None]:
        method_name = "validate_command" if operation == "command" else "validate_query"
        validate_method = getattr(message, method_name, None)
        if callable(validate_method):
            result = validate_method()
            if hasattr(result, "success") and hasattr(result, "error"):
                return cast("FlextResult[None]", result)
        return FlextResult[None].ok(None)

    def handle(self, message: MessageT) -> FlextResult[ResultT]:
        """Handle the message and return result.

        Subclasses must override this method.
        """
        msg = "Subclasses must implement handle method"
        raise NotImplementedError(msg)

    def execute(self, command: MessageT) -> FlextResult[ResultT]:
        """Execute command with full validation and error handling."""
        return self._run_pipeline(command, operation="command")

    def handle_query(self, query: MessageT) -> FlextResult[ResultT]:
        """Execute query with validation and error handling."""
        return self._run_pipeline(query, operation="query")

    def handle_command(self, command: MessageT) -> FlextResult[ResultT]:
        """Alias for execute to maintain semantic clarity."""
        return self.execute(command)

    def _run_pipeline(
        self,
        message: MessageT,
        *,
        operation: Literal["command", "query"],
    ) -> FlextResult[ResultT]:
        if self.mode != operation:
            error_msg = (
                f"{self.handler_name} is configured for {self.mode} operations "
                f"and cannot execute {operation} pipelines"
            )
            self.logger.error("invalid_handler_mode", error_message=error_msg)
            return FlextResult[ResultT].fail(
                error_msg,
                error_code=FlextConstants.Errors.COMMAND_HANDLER_NOT_FOUND,
            )

        message_type = type(message).__name__
        identifier = getattr(
            message,
            "command_id" if operation == "command" else "query_id",
            getattr(message, "id", "unknown"),
        )

        self.logger.info(
            "starting_handler_pipeline",
            handler_mode=self.mode,
            message_type=message_type,
            message_id=identifier,
        )

        # Check if handler can handle this message type
        if not self.can_handle(type(message)):
            error_msg = f"{self.handler_name} cannot handle {message_type}"
            self.logger.error("handler_cannot_handle", error_message=error_msg)
            return FlextResult[ResultT].fail(
                error_msg,
                error_code=FlextConstants.Errors.COMMAND_HANDLER_NOT_FOUND,
            )

        validation_result = self._validate_message(message, operation=operation)
        if validation_result.is_failure:
            self.logger.info(
                "handler_validation_failed",
                handler_mode=self.mode,
                message_type=message_type,
                error=validation_result.error,
            )
            return FlextResult[ResultT].fail(
                validation_result.error or "Validation failed",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

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

        except (TypeError, ValueError, AttributeError, RuntimeError) as exc:
            elapsed = time.time() - (self._start_time or 0.0)
            execution_time_ms = round(elapsed * 1000, 2) if elapsed else 0
            self.logger.exception(
                "handler_pipeline_failed",
                handler_mode=self.mode,
                message_type=message_type,
                message_id=identifier,
                execution_time_ms=execution_time_ms,
            )
            return FlextResult[ResultT].fail(
                f"Handler processing failed: {exc}",
                error_code=FlextConstants.Errors.COMMAND_PROCESSING_FAILED,
            )


__all__: FlextTypes.Core.StringList = [
    "FlextHandlers",
]
