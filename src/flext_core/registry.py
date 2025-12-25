"""Handler registration and discovery utilities.

FlextRegistry wires handlers to ``FlextDispatcher`` with explicit binding,
idempotent tracking, and batch registration support that matches the current
dispatcher-centric application layer.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import sys
from typing import Annotated, Self

from pydantic import Field, computed_field

from flext_core.constants import c
from flext_core.dispatcher import FlextDispatcher
from flext_core.handlers import (
    FlextHandlers,
)

# Local import to avoid circular dependency
from flext_core.mixins import FlextMixins as x
from flext_core.models import m
from flext_core.protocols import p
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.typings import t
from flext_core.utilities import u


class FlextRegistry(x):
    """Application-layer registry for CQRS handlers.

    The registry pairs message types with handlers, enforces idempotent
    registration, and exposes batch operations that return ``r``
    summaries. It delegates to ``FlextDispatcher`` (which implements
    ``p.CommandBus``) for actual handler registration and execution.
    """

    class Summary(m.Value):
        """Aggregated outcome for batch handler registration tracking.

        Provides comprehensive summary of batch handler registration operations,
        tracking successful, skipped, and failed registrations with detailed
        metadata and computed success indicators.

        This immutable value object is used by FlextRegistry to report the
        results of batch handler registration operations, enabling idempotent
        registration with full auditability.

        Attributes:
            registered: List of successfully registered handlers with details
            skipped: List of handler identifiers that were skipped (already registered)
            errors: List of error messages for failed registrations

        Computed Properties:
            is_success: True if no errors occurred during registration
            successful_registrations: Count of successfully registered handlers
            failed_registrations: Count of failed registration attempts

        Examples:
            >>> from pydantic import Field
            >>> from flext_core import FlextConstants
            >>> summary = FlextRegistry.Summary(
            ...     registered=[
            ...         m.HandlerRegistrationDetails(
            ...             registration_id="reg-001",
            ...             handler_mode=c.Cqrs.HandlerType.COMMAND,
            ...             timestamp="2025-01-01T00:00:00Z",
            ...             status=c.Cqrs.CommonStatus.RUNNING,
            ...         )
            ...     ],
            ...     skipped=["CreateUserCommand"],
            ...     errors=[],
            ... )
            >>> summary.is_success
            True
            >>> summary.successful_registrations
            1

        """

        registered: Annotated[
            list[m.HandlerRegistrationDetails],
            Field(
                default_factory=list,
                description="Successfully registered handlers with registration details",
            ),
        ] = Field(default_factory=list)
        skipped: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Handler identifiers that were skipped (already registered)",
                examples=[["CreateUserCommand", "UpdateUserCommand"]],
            ),
        ] = Field(default_factory=list)
        errors: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Error messages for failed registrations",
                examples=[["Handler validation failed", "Duplicate registration"]],
            ),
        ] = Field(default_factory=list)

        @computed_field
        def is_success(self) -> bool:
            """Indicate whether the batch registration fully succeeded.

            Returns:
                True if no errors occurred, False otherwise

            Examples:
                >>> summary = FlextRegistry.Summary(registered=[...], errors=[])
                >>> summary.is_success
                True

            """
            return not self.errors

        @computed_field
        def is_failure(self) -> bool:
            """Indicate whether the batch registration had errors.

            Returns:
                True if any errors occurred, False otherwise

            Examples:
                >>> summary = FlextRegistry.Summary(errors=["error1"])
                >>> summary.is_failure
                True

            """
            return bool(self.errors)

        @computed_field
        def successful_registrations(self) -> int:
            """Number of successful registrations.

            Returns:
                Count of successfully registered handlers

            Examples:
                >>> summary = FlextRegistry.Summary(registered=[detail1, detail2])
                >>> summary.successful_registrations
                2

            """
            return len(self.registered)

        @computed_field
        def failed_registrations(self) -> int:
            """Number of failed registrations.

            Returns:
                Count of failed registration attempts

            Examples:
                >>> summary = FlextRegistry.Summary(errors=["error1", "error2"])
                >>> summary.failed_registrations
                2

            """
            return len(self.errors)

        def __bool__(self) -> bool:
            """Boolean representation - False when there are errors, True otherwise.

            Returns:
                False if any errors occurred (is_failure), True if successful

            """
            return not self.errors

    def __init__(self, dispatcher: p.CommandBus | None = None) -> None:
        """Initialize the registry with a CommandBus protocol instance.

        Args:
            dispatcher: CommandBus protocol instance (defaults to creating FlextDispatcher)

        """
        super().__init__()

        # Initialize service infrastructure (DI, Context, Logging, Metrics)
        self._init_service("flext_registry")

        # Structural typing - FlextDispatcher implements p.CommandBus
        # Create dispatcher instance if not provided
        actual_dispatcher: p.CommandBus = (
            dispatcher
            if dispatcher is not None
            else FlextDispatcher()
        )
        self._dispatcher: p.CommandBus = actual_dispatcher

        # Enrich context with registry metadata for observability
        self._enrich_context(
            service_type="registry",
            dispatcher_type=type(dispatcher).__name__,
            supports_batch_registration=True,
            idempotent_registration=True,
        )
        self._registered_keys: set[str] = set()

    # ------------------------------------------------------------------
    # Factory Method with Auto-Discovery
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        dispatcher: p.CommandBus | None = None,
        *,
        auto_discover_handlers: bool = False,
    ) -> Self:
        """Factory method to create a new FlextRegistry instance.

        This is the preferred way to instantiate FlextRegistry. It provides
        a clean factory pattern that each class owns, respecting Clean
        Architecture principles where higher layers create their own instances.

        Auto-discovery of handlers discovers all functions marked with
        @FlextHandlers.handler() decorator in the calling module and auto-registers them
        with built-in deduplication. This enables zero-config handler
        registration for services with idempotent tracking.

        Args:
            dispatcher: Optional CommandBus instance (defaults to FlextDispatcher)
            auto_discover_handlers: If True, scan calling module for @handler()
                decorated functions and auto-register them with deduplication.
                Default: False.

        Returns:
            FlextRegistry instance with auto-discovered handlers if enabled.

        Example:
            >>> registry = FlextRegistry.create(auto_discover_handlers=True)
            >>> result = registry.register_handler(create_user_handler)

        """
        instance = cls(dispatcher)

        if auto_discover_handlers:
            # Get the caller's frame to discover handlers in calling module
            frame = inspect.currentframe()
            if frame and frame.f_back:
                caller_globals = frame.f_back.f_globals
                # Get module name from globals
                module_name = caller_globals.get("__name__", "__main__")
                # Get module object from globals
                caller_module = sys.modules.get(module_name)
                if caller_module:
                    # Scan module for handler-decorated functions
                    handlers = FlextHandlers.Discovery.scan_module(caller_module)
                    for _handler_name, handler_func, _handler_config in handlers:
                        # Get actual handler from module
                        # Check if handler_func is not None before checking callable
                        if handler_func is not None and callable(handler_func):
                            # Register handler with deduplication built-in
                            # Deduplication happens in register_handler() via _registered_keys
                            # Cast handler_func to expected type for register_handler
                            # register_handler expects FlextHandlers[T, R] | None where FlextHandlers is from flext_core.handlers
                            handler_typed: (
                                FlextHandlers[t.GeneralValueType, t.GeneralValueType]
                                | None
                            ) = cast(
                                "FlextHandlers[t.GeneralValueType, t.GeneralValueType] | None",
                                handler_func,
                            )
                            _ = instance.register_handler(handler_typed)

        return instance

    @staticmethod
    def _safe_get_handler_mode(
        value: t.GeneralValueType,
    ) -> c.Cqrs.HandlerType:
        """Safely extract and validate handler mode from t.GeneralValueType value."""
        # Use u.Parser.parse() for cleaner enum parsing
        parse_result = u.Parser.parse(
            value,
            c.Cqrs.HandlerType,
            default=c.Cqrs.HandlerType.COMMAND,
            case_insensitive=True,
        )
        return (
            parse_result.value
            if parse_result.is_success
            else c.Cqrs.HandlerType.COMMAND
        )

    @staticmethod
    def _safe_get_status(
        value: t.GeneralValueType,
    ) -> c.Cqrs.CommonStatus:
        """Safely extract and validate status from t.GeneralValueType value."""
        # Handle special case: RegistrationStatus.ACTIVE -> CommonStatus.RUNNING
        if value == c.Cqrs.RegistrationStatus.ACTIVE:
            return c.Cqrs.CommonStatus.RUNNING
        if value == c.Cqrs.RegistrationStatus.INACTIVE:
            return c.Cqrs.CommonStatus.FAILED
        # Use u.Parser.parse() for cleaner enum parsing
        parse_result = u.Parser.parse(
            value,
            c.Cqrs.CommonStatus,
            default=c.Cqrs.CommonStatus.RUNNING,
            case_insensitive=True,
        )
        return (
            parse_result.value
            if parse_result.is_success
            else c.Cqrs.CommonStatus.RUNNING
        )

    def _create_registration_details(
        self,
        reg_data: t.ConfigurationMapping,
        key: str,
    ) -> m.HandlerRegistrationDetails:
        """Create RegistrationDetails from registration data (DRY helper).

        Eliminates duplication in _process_single_handler and register_handler_batch.
        Both methods create RegistrationDetails from dict in identical way.

        Args:
            reg_data: Registration data dict from dispatcher
            key: Handler key for registration_id

        Returns:
            RegistrationDetails: Validated registration details model

        """
        # Extract values using u.extract() for cleaner code
        registration_id_result = u.Mapper.extract(
            reg_data,
            "registration_id",
            default=key,
            required=False,
        )
        handler_mode_result = u.Mapper.extract(
            reg_data,
            "handler_mode",
            default=c.Dispatcher.HANDLER_MODE_COMMAND,
            required=False,
        )
        timestamp_result = u.Mapper.extract(
            reg_data,
            "timestamp",
            default="",
            required=False,
        )
        status_result = u.Mapper.extract(
            reg_data,
            "status",
            default=c.Dispatcher.REGISTRATION_STATUS_ACTIVE,
            required=False,
        )
        # Safe value extraction - handle failures gracefully
        registration_id = (
            str(registration_id_result.value)
            if registration_id_result.is_success
            else key
        )
        handler_mode = (
            handler_mode_result.value
            if handler_mode_result.is_success
            else c.Dispatcher.HANDLER_MODE_COMMAND
        )
        timestamp = (
            str(timestamp_result.value)
            if timestamp_result.is_success and timestamp_result.value
            else ""
        )
        status = (
            status_result.value
            if status_result.is_success
            else c.Dispatcher.REGISTRATION_STATUS_ACTIVE
        )
        return m.HandlerRegistrationDetails(
            registration_id=registration_id,
            handler_mode=FlextRegistry._safe_get_handler_mode(handler_mode),
            timestamp=timestamp,
            status=self._safe_get_status(status),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_handler(
        self,
        handler: FlextHandlers[t.GeneralValueType, t.GeneralValueType] | None,
    ) -> r[m.HandlerRegistrationDetails]:
        """Register an already-constructed handler instance.

        Re-registration is ignored and treated as success to guarantee
        idempotent behaviour when multiple packages attempt to register
        the same handler.

        Returns:
            r[FlextDispatcher.Registration[MessageT, ResultT]]: Success result with registration details.

        """
        # Propagate context for distributed tracing
        handler_name = handler.__class__.__name__ if handler else "unknown"
        self._propagate_context(f"register_handler_{handler_name}")

        self.logger.debug(
            "Starting handler registration",
            operation="register_handler",
            handler_name=handler_name,
            handler_type=type(handler).__name__ if handler else "None",
        )

        # Validate handler is not None
        if handler is None:
            self.logger.error(
                "FAILED: Handler is None - REGISTRATION ABORTED",
                operation="register_handler",
                consequence="Cannot register None handler",
                resolution_hint="Provide a valid handler instance",
            )
            return r[m.HandlerRegistrationDetails].fail(
                "Handler cannot be None",
            )

        key = FlextRegistry._resolve_handler_key(handler)
        if key in self._registered_keys:
            self.logger.debug(
                "Handler already registered, returning existing registration",
                operation="register_handler",
                handler_name=handler_name,
                handler_key=key,
            )
            # Return successful registration details for idempotent registration
            return r[m.HandlerRegistrationDetails].ok(
                m.HandlerRegistrationDetails(
                    registration_id=key,
                    handler_mode=c.Cqrs.HandlerType.COMMAND,
                    timestamp="",  # Will be set by model if needed
                    status=c.Cqrs.CommonStatus.RUNNING,
                ),
            )

        # Handler is already the correct type
        self.logger.debug(
            "Registering handler with dispatcher",
            operation="register_handler",
            handler_name=handler_name,
            handler_key=key,
        )
        # register_handler returns r[t.ConfigurationDict]
        # register_handler accepts t.GeneralValueType | BaseModel, but h works via runtime check
        # Cast handler to t.GeneralValueType for type compatibility (runtime handles h correctly)
        registration = self._dispatcher.register_handler(
            cast("t.GeneralValueType", handler),
        )
        if registration.is_success:
            self._registered_keys.add(key)
            # Use get() for concise dict extraction
            reg_dict = registration.value
            _ = (
                u.Mapper.get(reg_dict, "handler_name", default=handler_name)
                or handler_name
            )
            self.logger.info(
                "Handler registered successfully",
                operation="register_handler",
                handler_name=handler_name,
                handler_key=key,
            )
            # Convert t.GeneralValueType to RegistrationDetails using helper method
            reg_data = registration.value
            reg_details = self._create_registration_details(reg_data, key)
            # Override timestamp with formatted default if not provided
            if not reg_details.timestamp:
                reg_details.timestamp = u.Generators.generate_iso_timestamp().replace(
                    "+00:00",
                    "Z",
                )
            return r[m.HandlerRegistrationDetails].ok(reg_details)

        error_msg = registration.error or "Unknown error"
        self.logger.error(
            "FAILED to register handler with dispatcher - REGISTRATION ABORTED",
            operation="register_handler",
            handler_name=handler_name,
            handler_key=key,
            error=error_msg,
            consequence="Handler will not be available for dispatch",
            resolution_hint="Check dispatcher configuration and handler compatibility",
        )
        return r[m.HandlerRegistrationDetails].fail(error_msg)

    def register_handlers(
        self,
        handlers: Iterable[FlextHandlers[t.GeneralValueType, t.GeneralValueType]],
    ) -> r[FlextRegistry.Summary]:
        """Register multiple handlers in one shot using railway pattern.

        Returns:
            r[FlextRegistry.Summary]: Success result with registration summary.

        """
        # Propagate context for distributed tracing
        self._propagate_context("register_handlers_batch")

        handlers_list = list(handlers)
        self.logger.info(
            "Starting batch handler registration",
            operation="register_handlers",
            handlers_count=len(handlers_list),
        )

        summary = FlextRegistry.Summary()
        for idx, handler in enumerate(handlers_list):
            handler_name = handler.__class__.__name__ if handler else "unknown"
            self.logger.debug(
                "Processing handler in batch",
                operation="register_handlers",
                handler_index=idx + 1,
                total_handlers=len(handlers_list),
                handler_name=handler_name,
            )
            result: r[bool] = self._process_single_handler(handler, summary)
            if result.is_failure:
                # When is_failure is True, error is never None (fail() converts None to "")
                # Use error or fallback message
                base_msg = "Handler processing failed"
                error_msg = result.error or f"{base_msg} (operation failed)"
                if error_msg == f"{base_msg} (operation failed)":
                    # If error is empty string, use base message
                    error_msg = base_msg
                else:
                    error_msg = f"{base_msg}: {error_msg}"
                self.logger.error(
                    "FAILED: Batch registration stopped due to handler error",
                    operation="register_handlers",
                    handler_index=idx + 1,
                    total_handlers=len(handlers_list),
                    handler_name=handler_name,
                    error=error_msg,
                    successful_registrations=len(summary.registered),
                    skipped_registrations=len(summary.skipped),
                    consequence="Remaining handlers in batch will not be registered",
                )
                return r[FlextRegistry.Summary].fail(error_msg)

        final_summary = self._finalize_summary(summary)
        if final_summary.is_success:
            self.logger.info(
                "Batch handler registration completed successfully",
                operation="register_handlers",
                total_handlers=len(handlers_list),
                successful_registrations=len(summary.registered),
                skipped_registrations=len(summary.skipped),
            )
        else:
            self.logger.warning(
                "Batch handler registration completed with errors",
                operation="register_handlers",
                total_handlers=len(handlers_list),
                successful_registrations=len(summary.registered),
                failed_registrations=len(summary.errors),
                errors=summary.errors[:5],  # Show first 5 errors
            )
        return final_summary

    def _process_single_handler(
        self,
        handler: FlextHandlers[t.GeneralValueType, t.GeneralValueType],
        summary: FlextRegistry.Summary,
    ) -> r[bool]:
        """Process a single handler registration.

        Returns:
            r[bool]: Success with True if registration succeeds, failure with error details.

        """
        key = FlextRegistry._resolve_handler_key(handler)
        handler_name = handler.__class__.__name__ if handler else "unknown"

        if key in self._registered_keys:
            self.logger.debug(
                "Handler already registered, skipping",
                operation="process_single_handler",
                handler_name=handler_name,
                handler_key=key,
            )
            summary.skipped.append(key)
            return r[bool].ok(True)

        # Handler is already the correct type
        self.logger.debug(
            "Registering handler with dispatcher",
            operation="process_single_handler",
            handler_name=handler_name,
            handler_key=key,
        )
        # register_handler accepts t.GeneralValueType | BaseModel, but h works via runtime check
        registration_result = self._dispatcher.register_handler(
            cast("t.GeneralValueType", handler),
        )
        if registration_result.is_success:
            # Convert dict result to RegistrationDetails
            reg_data = registration_result.value
            reg_details = self._create_registration_details(reg_data, key)
            if not reg_details:
                # Fallback: create minimal registration details
                reg_details = m.HandlerRegistrationDetails(
                    registration_id=key,
                    handler_mode=c.Cqrs.HandlerType.COMMAND,
                    timestamp="",
                    status=c.Cqrs.CommonStatus.RUNNING,
                )
            self._add_successful_registration(key, reg_details, summary)
            self.logger.debug(
                "Handler registered successfully in batch",
                operation="process_single_handler",
                handler_name=handler_name,
                handler_key=key,
            )
            return r[bool].ok(True)
        error_msg = registration_result.error
        self.logger.warning(
            "Handler registration failed in batch",
            operation="process_single_handler",
            handler_name=handler_name,
            handler_key=key,
            error=error_msg,
            consequence="Handler will not be available for dispatch",
        )
        # Use error property for type-safe str
        error_str = registration_result.error or "Unknown error"
        _ = FlextRegistry._add_registration_error(key, error_str, summary)
        # Return type should be r[bool] but we're returning r[Summary]
        # Fix: return r[bool] as expected by method signature
        return r[bool].fail(error_str)

    def _add_successful_registration(
        self,
        key: str,
        registration: m.HandlerRegistrationDetails,
        summary: FlextRegistry.Summary,
    ) -> None:
        """Add successful registration to summary."""
        self._registered_keys.add(key)
        summary.registered.append(
            registration,
        )

    @staticmethod
    def _add_registration_error(
        key: str,
        error: str,
        summary: FlextRegistry.Summary,
    ) -> str:
        """Add registration error to summary.

        Returns:
            str: The error message that was added.

        """
        # When is_failure is True, error is never None (fail() converts None to "")
        # Use error or fallback message
        summary.errors.append(error or f"Failed to register handler '{key}'")
        return error

    def _finalize_summary(
        self,
        summary: FlextRegistry.Summary,
    ) -> r[FlextRegistry.Summary]:
        """Finalize summary based on error state.

        Returns:
            r[FlextRegistry.Summary]: Success result with summary or failure result with errors.

        """
        if summary.errors:
            return r[FlextRegistry.Summary].fail(
                "; ".join(summary.errors),
            )
        return r[FlextRegistry.Summary].ok(summary)

    def register_bindings(
        self,
        bindings: Sequence[
            tuple[
                type[t.GeneralValueType],
                FlextHandlers[t.GeneralValueType, t.GeneralValueType],
            ]
        ],
    ) -> r[FlextRegistry.Summary]:
        """Register handlers bound to explicit message types.

        Returns:
            r[FlextRegistry.Summary]: Success result with registration summary.

        """
        summary = FlextRegistry.Summary()
        for message_type, handler in bindings:
            key = FlextRegistry._resolve_binding_key(handler, message_type)
            if key in self._registered_keys:
                summary.skipped.append(key)
                continue

            # Structural typing - handler is h which implements p.Handler
            # Cast to t.GeneralValueType for protocol compatibility
            registration = self._dispatcher.register_command(
                message_type,
                cast("t.GeneralValueType", handler),
            )
            if registration.is_failure:
                # When is_failure is True, error is never None (fail() converts None to "")
                # Use error or fallback message
                summary.errors.append(
                    registration.error
                    or f"Failed to register handler '{key}' for '{message_type.__name__}'",
                )
                continue

            self._registered_keys.add(key)
            # Type narrowing: registration.value can be various types, ensure it's dict
            reg_data_raw = registration.value
            if isinstance(reg_data_raw, dict):
                # Type narrowing: reg_data_raw is dict, use directly
                reg_data_dict: t.ConfigurationMapping = reg_data_raw
                reg_details = self._create_registration_details(
                    reg_data_dict,
                    key,
                )
            elif isinstance(reg_data_raw, Mapping):
                # Type narrowing: reg_data_raw is Mapping (but not dict), convert to dict
                reg_data_dict = u.mapper().to_dict(reg_data_raw)
                reg_details = self._create_registration_details(
                    reg_data_dict,
                    key,
                )
            else:
                # Not a mapping, create basic details
                reg_details = m.HandlerRegistrationDetails(
                    registration_id=key,
                    handler_mode=c.Cqrs.HandlerType.COMMAND,
                    timestamp="",
                    status=c.Cqrs.CommonStatus.RUNNING,
                )
            summary.registered.append(reg_details)

        if summary.errors:
            return r[FlextRegistry.Summary].fail(
                "; ".join(summary.errors),
            )
        return r[FlextRegistry.Summary].ok(summary)

    def register_function_map(
        self,
        mapping: Mapping[
            type[t.GeneralValueType],
            (
                FlextHandlers[t.GeneralValueType, t.GeneralValueType]
                | tuple[
                    t.HandlerCallable,
                    t.GeneralValueType | r[t.GeneralValueType],
                ]
                | t.GeneralValueType
                | tuple[t.GeneralValueType, ...]
                | None
            ),
        ],
    ) -> r[FlextRegistry.Summary]:
        """Register plain callables or pre-built handlers for message types.

        Refactored with DRY helpers to reduce nesting (6 → 3 levels).

        Returns:
            r[FlextRegistry.Summary]: Success result with registration summary.

        """
        summary = FlextRegistry.Summary()
        for message_type, entry in mapping.items():
            try:
                # Resolve key and check if already registered (early continue)
                key = FlextRegistry._resolve_binding_key_from_entry(entry, message_type)
                if key in self._registered_keys:
                    summary.skipped.append(key)
                    continue

                # Delegate to specialized helpers based on entry type
                # Use guard() with tuple type and length check
                def tuple_check(v: object) -> bool:
                    return (
                        isinstance(v, tuple)
                        and len(v) == c.Performance.EXPECTED_TUPLE_LENGTH
                    )

                tuple_result = u.Validation.guard(
                    entry,
                    tuple,
                    tuple_check,
                    return_value=True,
                )
                if tuple_result and isinstance(tuple_result, tuple):
                    handler_elem, config_elem = tuple_result
                    # Type narrowing: handler_elem should be HandlerCallableType
                    if isinstance(handler_elem, type) or callable(handler_elem):
                        # Cast handler_elem to HandlerCallable for type safety
                        handler_typed: t.HandlerCallable = cast(
                            "t.HandlerCallable",
                            handler_elem,
                        )
                        tuple_entry: tuple[
                            t.HandlerCallable,
                            t.GeneralValueType | r[t.GeneralValueType],
                        ] = (handler_typed, config_elem)
                        result = self._register_tuple_entry(tuple_entry, key)
                    else:
                        result = FlextRegistry._register_other_entry(key)
                elif isinstance(entry, FlextHandlers):
                    # Type narrowing: entry is FlextHandlers instance
                    result = self._register_handler_entry(entry, key)
                else:
                    result = self._register_other_entry(key)

                # Process result (nesting reduced via early returns in helpers)
                if result.is_success:
                    summary.registered.append(result.value)
                    self._registered_keys.add(key)
                else:
                    # When is_failure is True, error is never None (fail() converts None to "")
                    # Use error or empty string as fallback
                    summary.errors.append(result.error or "")

            except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
                error_msg = f"Failed to register handler for {message_type}: {e}"
                summary.errors.append(error_msg)
                continue

        return r[FlextRegistry.Summary].ok(summary)

    # ------------------------------------------------------------------
    # Internal helpers for register_function_map (DRY + Reduce nesting)
    # ------------------------------------------------------------------
    def _register_tuple_entry(
        self,
        entry: tuple[
            t.HandlerCallable,
            t.GeneralValueType | r[t.GeneralValueType],
        ],
        key: str,
    ) -> r[m.HandlerRegistrationDetails]:
        """Register tuple (function, config) entry - DRY helper reduces nesting."""
        handler_func, handler_config = entry

        # Use guard with return_value=True for concise dict validation
        config_dict = (
            u.Validation.guard(handler_config, dict, return_value=True)
            if handler_config is not None
            else None
        )
        # Cast config_dict to HandlerConfigurationType
        handler_config_typed: t.ConfigurationMapping | None = cast(
            "t.ConfigurationMapping | None",
            config_dict,
        )
        # Structural typing - handler_func is HandlerCallable compatible
        # Cast handler_config to protocol-compatible type
        handler_result = self._dispatcher.create_handler_from_function(
            handler_func,
            handler_config=cast(
                "dict[str, t.FlexibleValue] | None",
                handler_config_typed,
            ),
            mode=c.Cqrs.HandlerType.COMMAND,
        )
        if handler_result.is_failure:
            return r[m.HandlerRegistrationDetails].fail(
                f"Failed to create handler: {handler_result.error}",
            )

        # Register with dispatcher
        handler = handler_result.value
        # register_handler accepts t.GeneralValueType | BaseModel, but h works via runtime check
        register_result = self._dispatcher.register_handler(
            cast("t.GeneralValueType", handler),
        )
        if register_result.is_failure:
            return r[m.HandlerRegistrationDetails].fail(
                f"Failed to register handler: {register_result.error}",
            )

        # Success - create registration details
        reg_details = m.HandlerRegistrationDetails(
            registration_id=key,
            handler_mode=c.Cqrs.HandlerType.COMMAND,
            timestamp="",
            status=c.Cqrs.CommonStatus.RUNNING,
        )
        return r[m.HandlerRegistrationDetails].ok(reg_details)

    def _register_handler_entry(
        self,
        entry: FlextHandlers[t.GeneralValueType, t.GeneralValueType],
        key: str,
    ) -> r[m.HandlerRegistrationDetails]:
        """Register FlextHandlers instance - DRY helper reduces nesting."""
        # register_handler accepts t.GeneralValueType | BaseModel, but h works via runtime check
        register_result = self._dispatcher.register_handler(
            cast("t.GeneralValueType", entry),
        )
        if register_result.is_failure:
            return r[m.HandlerRegistrationDetails].fail(
                f"Failed to register handler: {register_result.error}",
            )

        # Success - create registration details
        reg_details = m.HandlerRegistrationDetails(
            registration_id=key,
            handler_mode=c.Cqrs.HandlerType.COMMAND,
            timestamp="",
            status=c.Cqrs.CommonStatus.RUNNING,
        )
        return r[m.HandlerRegistrationDetails].ok(reg_details)

    @staticmethod
    def _register_other_entry(
        key: str,
    ) -> r[m.HandlerRegistrationDetails]:
        """Register t.GeneralValueType or other types - DRY helper reduces nesting."""
        reg_details = m.HandlerRegistrationDetails(
            registration_id=key,
            handler_mode=c.Cqrs.HandlerType.COMMAND,
            timestamp="",
            status=c.Cqrs.CommonStatus.RUNNING,
        )
        return r[m.HandlerRegistrationDetails].ok(reg_details)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_handler_key(
        handler: FlextHandlers[t.GeneralValueType, t.GeneralValueType],
    ) -> str:
        handler_id = getattr(handler, "handler_id", None)
        return (
            handler_id
            if handler_id and u.is_type(handler_id, str)
            else handler.__class__.__name__
        )

    @staticmethod
    def _resolve_binding_key(
        handler: FlextHandlers[t.GeneralValueType, t.GeneralValueType],
        message_type: type[t.GeneralValueType],
    ) -> str:
        base_key = FlextRegistry._resolve_handler_key(handler)
        # Handle both type objects and string keys
        if hasattr(message_type, "__name__"):
            type_name = message_type.__name__
        else:
            type_name = str(message_type)
        return f"{base_key}::{type_name}"

    @staticmethod
    def _resolve_binding_key_from_entry(
        entry: (
            FlextHandlers[t.GeneralValueType, t.GeneralValueType]
            | tuple[
                t.HandlerCallable,
                t.GeneralValueType | r[t.GeneralValueType],
            ]
            | t.GeneralValueType
            | tuple[t.GeneralValueType, ...]
            | None
        ),
        message_type: type[t.GeneralValueType],
    ) -> str:
        # Use guard() with tuple type and length check
        def tuple_check(v: object) -> bool:
            return (
                isinstance(v, tuple) and len(v) == c.Performance.EXPECTED_TUPLE_LENGTH
            )

        tuple_result = u.Validation.guard(entry, tuple, tuple_check, return_value=True)
        if tuple_result and isinstance(tuple_result, tuple):
            handler_func = tuple_result[0]
            handler_name = getattr(handler_func, "__name__", None) or str(handler_func)
            # Handle both type objects and string keys
            if hasattr(message_type, "__name__"):
                type_name = message_type.__name__
            else:
                type_name = str(message_type)
            return f"{handler_name}::{type_name}"
        if isinstance(entry, FlextHandlers):
            # Type narrowing: entry is FlextHandlers instance
            return FlextRegistry._resolve_binding_key(entry, message_type)
        # Handle t.GeneralValueType or other types
        return str(entry)

    def register(
        self,
        name: str,
        service: t.GeneralValueType,
        metadata: t.GeneralValueType | m.Metadata | None = None,
    ) -> r[bool]:
        """Register a service component with optional metadata.

        Delegates to the container's register method for dependency injection.
        Metadata is currently stored for future use but not actively used.

        Args:
            name: Service name for later retrieval
            service: Service instance to register
            metadata: Optional metadata (dict or m.Metadata)

        Returns:
            r[bool]: Success (True) if registered or failure with error details.

        """
        # Normalize metadata to dict for internal use
        validated_metadata: t.GeneralValueType | None = None
        if metadata is not None:
            # Handle Metadata model first
            if isinstance(metadata, m.Metadata):
                validated_metadata = metadata.attributes
            else:
                # Cast to t.GeneralValueType for is_dict_like check
                metadata_as_general = metadata
                if FlextRuntime.is_dict_like(metadata_as_general):
                    # Type guard ensures metadata_as_general is t.ConfigurationMapping
                    # Cast to Mapping[str, T] for to_dict() call
                    metadata_mapping: t.ConfigurationMapping = cast(
                        "t.ConfigurationMapping",
                        metadata_as_general,
                    )
                    validated_metadata = u.mapper().to_dict(metadata_mapping)
                else:
                    return r[bool].fail(
                        f"metadata must be dict or m.Metadata, got {type(metadata).__name__}",
                    )

        # Store metadata if provided (for future use)
        if isinstance(validated_metadata, Mapping):
            # Type narrowing: validated_metadata is Mapping after isinstance check
            # Log metadata with service name for observability
            # Use guard with return_value=True and default for concise dict conversion
            metadata_dict_raw = u.Validation.guard(
                validated_metadata,
                dict,
                default=dict(validated_metadata.items()),
                return_value=True,
            )
            # Type narrowing: guard returns r[dict] | dict | None when return_value=True
            if isinstance(metadata_dict_raw, r):
                metadata_dict = (
                    metadata_dict_raw.value
                    if isinstance(metadata_dict_raw.value, dict)
                    else {}
                )
            elif isinstance(metadata_dict_raw, dict):
                metadata_dict = metadata_dict_raw
            else:
                metadata_dict = {}
            metadata_keys: list[str] = list(metadata_dict.keys())
            self.logger.debug(
                "Registering service with metadata",
                operation="with_service",
                service_name=name,
                has_metadata=True,
                metadata_keys=metadata_keys,
            )

        # Delegate to container (x.container returns FlextContainer)
        # Use with_service for fluent API compatibility (returns Self)
        try:
            # service is already t.GeneralValueType (from method signature)
            # with_service returns Self for fluent chaining, but we don't need the return value
            _ = self.container.with_service(name, service)
            return r[bool].ok(True)
        except ValueError as e:
            error_str = str(e)
            return r[bool].fail(error_str)

    # =========================================================================
    # Protocol Implementations: RegistrationTracker, BatchProcessor
    # =========================================================================

    def register_item(
        self,
        name: str,
        item: t.GeneralValueType,
    ) -> r[bool]:
        """Register item (RegistrationTracker protocol)."""
        # Direct implementation without try/except - use FlextResult for error handling
        return self.register(name, item)

    def get_item(self, name: str) -> r[t.GeneralValueType]:
        """Get registered item (RegistrationTracker protocol)."""
        try:
            return r[t.GeneralValueType].ok(getattr(self, name))
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return r[t.GeneralValueType].fail(str(e))

    def list_items(self) -> r[list[str]]:
        """List registered items (RegistrationTracker protocol)."""
        try:
            keys = list(getattr(self, "_registered_keys", []))
            return r[list[str]].ok(keys)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return r[list[str]].fail(str(e))

    @staticmethod
    def get_batch_size() -> int:
        """Get batch size (BatchProcessor protocol)."""
        return 100


__all__ = ["FlextRegistry"]
