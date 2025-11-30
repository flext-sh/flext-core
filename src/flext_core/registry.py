"""Handler registration and discovery utilities.

FlextRegistry wires handlers to ``FlextDispatcher`` with explicit binding,
idempotent tracking, and batch registration support that matches the current
dispatcher-centric application layer.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated

from pydantic import Field, computed_field

from flext_core.constants import FlextConstants
from flext_core.dispatcher import FlextDispatcher
from flext_core.handlers import FlextHandlers
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities


class FlextRegistry(FlextMixins):
    """Application-layer registry for CQRS handlers.

    The registry pairs message types with handlers, enforces idempotent
    registration, and exposes batch operations that return ``FlextResult``
    summaries. It satisfies ``FlextProtocols.HandlerRegistry`` through
    structural typing and integrates directly with ``FlextDispatcher``.
    """

    class Summary(FlextModels.Value):
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
            ...         FlextModels.RegistrationDetails(
            ...             registration_id="reg-001",
            ...             handler_mode=FlextConstants.Cqrs.HandlerType.COMMAND,
            ...             timestamp="2025-01-01T00:00:00Z",
            ...             status=FlextConstants.Cqrs.Status.RUNNING,
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
            list[FlextModels.RegistrationDetails],
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

    def __init__(self, dispatcher: FlextDispatcher) -> None:
        """Initialize the registry with a FlextDispatcher instance."""
        super().__init__()

        # Initialize service infrastructure (DI, Context, Logging, Metrics)
        self._init_service("flext_registry")

        # Enrich context with registry metadata for observability
        self._enrich_context(
            service_type="registry",
            dispatcher_type=type(dispatcher).__name__,
            supports_batch_registration=True,
            idempotent_registration=True,
        )

        self._dispatcher = dispatcher
        self._registered_keys: set[str] = set()

    def _safe_get_handler_mode(
        self,
        value: FlextTypes.GeneralValueType,
    ) -> FlextConstants.Cqrs.HandlerType:
        """Safely extract and validate handler mode from object value."""
        if value == FlextConstants.Cqrs.HandlerType.QUERY:
            return FlextConstants.Cqrs.QUERY_HANDLER_TYPE
        if value == FlextConstants.Cqrs.HandlerType.COMMAND:
            return FlextConstants.Cqrs.COMMAND_HANDLER_TYPE
        # Default to command for invalid values
        return FlextConstants.Cqrs.COMMAND_HANDLER_TYPE

    def _safe_get_status(
        self,
        value: FlextTypes.GeneralValueType,
    ) -> FlextConstants.Cqrs.Status:
        """Safely extract and validate status from object value."""
        if value == FlextConstants.Cqrs.RegistrationStatus.ACTIVE:
            return FlextConstants.Cqrs.Status.RUNNING
        if value == FlextConstants.Cqrs.RegistrationStatus.INACTIVE:
            return FlextConstants.Cqrs.Status.COMPLETED
        # Default to running for invalid values
        return FlextConstants.Cqrs.Status.RUNNING

    def _create_registration_details(
        self,
        reg_data: Mapping[str, FlextTypes.GeneralValueType],
        key: str,
    ) -> FlextModels.RegistrationDetails:
        """Create RegistrationDetails from registration data (DRY helper).

        Eliminates duplication in _process_single_handler and register_handler_batch.
        Both methods create RegistrationDetails from dict in identical way.

        Args:
            reg_data: Registration data dict from dispatcher
            key: Handler key for registration_id

        Returns:
            RegistrationDetails: Validated registration details model

        """
        return FlextModels.RegistrationDetails(
            registration_id=str(reg_data.get("registration_id", key)),
            handler_mode=self._safe_get_handler_mode(
                reg_data.get(
                    "handler_mode",
                    FlextConstants.Dispatcher.HANDLER_MODE_COMMAND,
                ),
            ),
            # Fast fail: timestamp must be str or None
            timestamp=(
                str(reg_data.get("timestamp"))
                if reg_data.get("timestamp") is not None
                else ""
            ),
            status=self._safe_get_status(
                reg_data.get(
                    "status",
                    FlextConstants.Dispatcher.REGISTRATION_STATUS_ACTIVE,
                ),
            ),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_handler(
        self,
        handler: FlextHandlers[object, object] | None,
    ) -> FlextResult[FlextModels.RegistrationDetails]:
        """Register an already-constructed handler instance.

        Re-registration is ignored and treated as success to guarantee
        idempotent behaviour when multiple packages attempt to register
        the same handler.

        Returns:
            FlextResult[FlextDispatcher.Registration[MessageT, ResultT]]: Success result with registration details.

        """
        # Propagate context for distributed tracing
        handler_name = handler.__class__.__name__ if handler else "unknown"
        self._propagate_context(f"register_handler_{handler_name}")

        self.logger.debug(
            "Starting handler registration",
            operation="register_handler",
            handler_name=handler_name,
            handler_type=type(handler).__name__ if handler else "None",
            source="flext-core/src/flext_core/registry.py",
        )

        # Validate handler is not None
        if handler is None:
            self.logger.error(
                "FAILED: Handler is None - REGISTRATION ABORTED",
                operation="register_handler",
                consequence="Cannot register None handler",
                resolution_hint="Provide a valid handler instance",
                source="flext-core/src/flext_core/registry.py",
            )
            return self.fail(
                "Handler cannot be None",
            )

        key = self._resolve_handler_key(handler)
        if key in self._registered_keys:
            self.logger.debug(
                "Handler already registered, returning existing registration",
                operation="register_handler",
                handler_name=handler_name,
                handler_key=key,
                source="flext-core/src/flext_core/registry.py",
            )
            # Return successful registration details for idempotent registration
            return self.ok(
                FlextModels.RegistrationDetails(
                    registration_id=key,
                    handler_mode=FlextConstants.Cqrs.COMMAND_HANDLER_TYPE,
                    timestamp="",  # Will be set by model if needed
                    status=FlextConstants.Cqrs.Status.RUNNING,
                ),
            )

        # Handler is already the correct type
        self.logger.debug(
            "Registering handler with dispatcher",
            operation="register_handler",
            handler_name=handler_name,
            handler_key=key,
            source="flext-core/src/flext_core/registry.py",
        )
        registration = self._dispatcher.register_handler(handler)
        if registration.is_success:
            self._registered_keys.add(key)
            self.logger.info(
                "Handler registered successfully",
                operation="register_handler",
                handler_name=handler_name,
                handler_key=key,
                source="flext-core/src/flext_core/registry.py",
            )
            # Convert object to RegistrationDetails
            reg_data = registration.value
            # Format timestamp without microseconds to match pattern
            default_timestamp = (
                FlextUtilities.Generators.generate_iso_timestamp().replace(
                    "+00:00",
                    "Z",
                )
            )
            # Get timestamp from dispatcher or use default - ensure it matches pattern
            # Fast fail: timestamp must be str or None
            timestamp_value = reg_data.get("timestamp")
            dispatcher_timestamp: str = (
                "" if timestamp_value is None else str(timestamp_value)
            )
            timestamp_value = (
                default_timestamp
                if not dispatcher_timestamp
                else str(dispatcher_timestamp)
            )
            reg_details = FlextModels.RegistrationDetails(
                registration_id=str(reg_data.get("registration_id", key)),
                handler_mode=self._safe_get_handler_mode(
                    reg_data.get(
                        "handler_mode",
                        FlextConstants.Dispatcher.HANDLER_MODE_COMMAND,
                    ),
                ),
                timestamp=timestamp_value,
                status=self._safe_get_status(
                    reg_data.get(
                        "status",
                        FlextConstants.Dispatcher.REGISTRATION_STATUS_ACTIVE,
                    ),
                ),
            )
            return self.ok(reg_details)

        error_msg = registration.error or "Unknown error"
        self.logger.error(
            "FAILED to register handler with dispatcher - REGISTRATION ABORTED",
            operation="register_handler",
            handler_name=handler_name,
            handler_key=key,
            error=error_msg,
            consequence="Handler will not be available for dispatch",
            resolution_hint="Check dispatcher configuration and handler compatibility",
            source="flext-core/src/flext_core/registry.py",
        )
        return self.fail(error_msg)

    def register_handlers(
        self,
        handlers: Iterable[FlextHandlers[object, object]],
    ) -> FlextResult[FlextRegistry.Summary]:
        """Register multiple handlers in one shot using railway pattern.

        Returns:
            FlextResult[FlextRegistry.Summary]: Success result with registration summary.

        """
        # Propagate context for distributed tracing
        self._propagate_context("register_handlers_batch")

        handlers_list = list(handlers)
        self.logger.info(
            "Starting batch handler registration",
            operation="register_handlers",
            handlers_count=len(handlers_list),
            source="flext-core/src/flext_core/registry.py",
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
                source="flext-core/src/flext_core/registry.py",
            )
            result: FlextResult[bool] = self._process_single_handler(handler, summary)
            if result.is_failure:
                base_msg = "Handler processing failed"
                error_msg = (
                    f"{base_msg}: {result.error}"
                    if result.error
                    else f"{base_msg} (operation failed)"
                )
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
                    source="flext-core/src/flext_core/registry.py",
                )
                return self.fail(error_msg)

        final_summary = self._finalize_summary(summary)
        if final_summary.is_success:
            self.logger.info(
                "Batch handler registration completed successfully",
                operation="register_handlers",
                total_handlers=len(handlers_list),
                successful_registrations=len(summary.registered),
                skipped_registrations=len(summary.skipped),
                source="flext-core/src/flext_core/registry.py",
            )
        else:
            self.logger.warning(
                "Batch handler registration completed with errors",
                operation="register_handlers",
                total_handlers=len(handlers_list),
                successful_registrations=len(summary.registered),
                failed_registrations=len(summary.errors),
                errors=summary.errors[:5],  # Show first 5 errors
                source="flext-core/src/flext_core/registry.py",
            )
        return final_summary

    def _process_single_handler(
        self,
        handler: FlextHandlers[object, object],
        summary: FlextRegistry.Summary,
    ) -> FlextResult[bool]:
        """Process a single handler registration.

        Returns:
            FlextResult[bool]: Success with True if registration succeeds, failure with error details.

        """
        key = self._resolve_handler_key(handler)
        handler_name = handler.__class__.__name__ if handler else "unknown"

        if key in self._registered_keys:
            self.logger.debug(
                "Handler already registered, skipping",
                operation="process_single_handler",
                handler_name=handler_name,
                handler_key=key,
                source="flext-core/src/flext_core/registry.py",
            )
            summary.skipped.append(key)
            return self.ok(True)

        # Handler is already the correct type
        self.logger.debug(
            "Registering handler with dispatcher",
            operation="process_single_handler",
            handler_name=handler_name,
            handler_key=key,
            source="flext-core/src/flext_core/registry.py",
        )
        registration_result: FlextResult[FlextTypes.GeneralValueType] = (
            self._dispatcher.register_handler(handler)
        )
        if registration_result.is_success:
            # Convert object to RegistrationDetails
            reg_data = registration_result.value
            reg_details = self._create_registration_details(reg_data, key)
            self._add_successful_registration(key, reg_details, summary)
            self.logger.debug(
                "Handler registered successfully in batch",
                operation="process_single_handler",
                handler_name=handler_name,
                handler_key=key,
                source="flext-core/src/flext_core/registry.py",
            )
            return self.ok(True)
        error_msg = registration_result.error
        self.logger.warning(
            "Handler registration failed in batch",
            operation="process_single_handler",
            handler_name=handler_name,
            handler_key=key,
            error=error_msg,
            consequence="Handler will not be available for dispatch",
            source="flext-core/src/flext_core/registry.py",
        )
        # Use unwrap_error() for type-safe str
        error_str = registration_result.error or "Unknown error"
        self._add_registration_error(key, error_str, summary)
        return self.fail(error_str)

    def _add_successful_registration(
        self,
        key: str,
        registration: FlextModels.RegistrationDetails,
        summary: FlextRegistry.Summary,
    ) -> None:
        """Add successful registration to summary."""
        self._registered_keys.add(key)
        summary.registered.append(
            registration,
        )

    def _add_registration_error(
        self,
        key: str,
        error: str,
        summary: FlextRegistry.Summary,
    ) -> str:
        """Add registration error to summary.

        Returns:
            str: The error message that was added.

        """
        summary.errors.append(str(error) or f"Failed to register handler '{key}'")
        return error

    def _finalize_summary(
        self,
        summary: FlextRegistry.Summary,
    ) -> FlextResult[FlextRegistry.Summary]:
        """Finalize summary based on error state.

        Returns:
            FlextResult[FlextRegistry.Summary]: Success result with summary or failure result with errors.

        """
        if summary.errors:
            return self.fail(
                "; ".join(summary.errors),
            )
        return self.ok(summary)

    def register_bindings(
        self,
        bindings: Sequence[tuple[type[object], FlextHandlers[object, object]]],
    ) -> FlextResult[FlextRegistry.Summary]:
        """Register handlers bound to explicit message types.

        Returns:
            FlextResult[FlextRegistry.Summary]: Success result with registration summary.

        """
        summary = FlextRegistry.Summary()
        for message_type, handler in bindings:
            key = self._resolve_binding_key(handler, message_type)
            if key in self._registered_keys:
                summary.skipped.append(key)
                continue

            # Handler is already the correct type
            registration = self._dispatcher.register_command(message_type, handler)
            if registration.is_failure:
                summary.errors.append(
                    str(registration.error)
                    or f"Failed to register handler '{key}' for '{message_type.__name__}'",
                )
                continue

            self._registered_keys.add(key)
            # Convert object to RegistrationDetails
            reg_data = registration.value
            reg_details = self._create_registration_details(reg_data, key)
            summary.registered.append(reg_details)

        if summary.errors:
            return self.fail(
                "; ".join(summary.errors),
            )
        return self.ok(summary)

    def register_function_map(
        self,
        mapping: Mapping[
            type[object],
            FlextHandlers[object, object]
            | tuple[
                FlextTypes.Bus.HandlerCallableType,
                object | FlextResult[FlextTypes.GeneralValueType],
            ]
            | object
            | tuple[object, ...]
            | None,
        ],
    ) -> FlextResult[FlextRegistry.Summary]:
        """Register plain callables or pre-built handlers for message types.

        Refactored with DRY helpers to reduce nesting (6 → 3 levels).

        Returns:
            FlextResult[FlextRegistry.Summary]: Success result with registration summary.

        """
        summary = FlextRegistry.Summary()
        for message_type, entry in mapping.items():
            try:
                # Resolve key and check if already registered (early continue)
                key = self._resolve_binding_key_from_entry(entry, message_type)
                if key in self._registered_keys:
                    summary.skipped.append(key)
                    continue

                # Delegate to specialized helpers based on entry type
                # Tuple entries must have exactly 2 elements: (handler, config)
                tuple_entry_size = 2
                if isinstance(entry, tuple) and len(entry) == tuple_entry_size:
                    # Type narrowing: verify it's a 2-tuple before passing
                    # Extract elements and pass to method
                    handler_elem = entry[0]
                    config_elem = entry[1]
                    # Type narrowing: handler_elem should be HandlerCallableType
                    if isinstance(handler_elem, type) or callable(handler_elem):
                        tuple_entry: tuple[
                            FlextTypes.Bus.HandlerCallableType,
                            object | FlextResult[FlextTypes.GeneralValueType],
                        ] = (handler_elem, config_elem)
                        result = self._register_tuple_entry(tuple_entry, key)
                    else:
                        result = self._register_other_entry(key)
                elif isinstance(entry, FlextHandlers):
                    result = self._register_handler_entry(entry, key)
                else:
                    result = self._register_other_entry(key)

                # Process result (nesting reduced via early returns in helpers)
                if result.is_success:
                    summary.registered.append(result.value)
                    self._registered_keys.add(key)
                elif result.error is not None:
                    summary.errors.append(result.error)

            except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
                error_msg = f"Failed to register handler for {message_type}: {e}"
                summary.errors.append(error_msg)
                continue

        return self.ok(summary)

    # ------------------------------------------------------------------
    # Internal helpers for register_function_map (DRY + Reduce nesting)
    # ------------------------------------------------------------------
    def _register_tuple_entry(
        self,
        entry: tuple[
            FlextTypes.Bus.HandlerCallableType,
            object | FlextResult[FlextTypes.GeneralValueType],
        ],
        key: str,
    ) -> FlextResult[FlextModels.RegistrationDetails]:
        """Register tuple (function, config) entry - DRY helper reduces nesting."""
        handler_func, handler_config = entry

        # Create handler from function
        # handler_config is object | FlextResult[FlextTypes.GeneralValueType], convert to object | None
        config_dict: FlextTypes.GeneralValueType | None = None
        if handler_config is not None and isinstance(handler_config, dict):
            config_dict = handler_config
            # If it's FlextResult, skip (not supported for config)
        handler_result = self._dispatcher.create_handler_from_function(
            handler_func,
            config_dict,
            FlextConstants.Cqrs.COMMAND_HANDLER_TYPE,
        )
        if handler_result.is_failure:
            return FlextResult[FlextModels.RegistrationDetails].fail(
                f"Failed to create handler: {handler_result.error}",
            )

        # Register with dispatcher
        handler = handler_result.value
        register_result = self._dispatcher.register_handler(handler)
        if register_result.is_failure:
            return FlextResult[FlextModels.RegistrationDetails].fail(
                f"Failed to register handler: {register_result.error}",
            )

        # Success - create registration details
        reg_details = FlextModels.RegistrationDetails(
            registration_id=key,
            handler_mode=FlextConstants.Cqrs.COMMAND_HANDLER_TYPE,
            timestamp="",
            status=FlextConstants.Cqrs.Status.RUNNING,
        )
        return FlextResult[FlextModels.RegistrationDetails].ok(reg_details)

    def _register_handler_entry(
        self,
        entry: FlextHandlers[object, object],
        key: str,
    ) -> FlextResult[FlextModels.RegistrationDetails]:
        """Register FlextHandlers instance - DRY helper reduces nesting."""
        register_result = self._dispatcher.register_handler(entry)
        if register_result.is_failure:
            return FlextResult[FlextModels.RegistrationDetails].fail(
                f"Failed to register handler: {register_result.error}",
            )

        # Success - create registration details
        reg_details = FlextModels.RegistrationDetails(
            registration_id=key,
            handler_mode=FlextConstants.Cqrs.COMMAND_HANDLER_TYPE,
            timestamp="",
            status=FlextConstants.Cqrs.Status.RUNNING,
        )
        return FlextResult[FlextModels.RegistrationDetails].ok(reg_details)

    def _register_other_entry(
        self,
        key: str,
    ) -> FlextResult[FlextModels.RegistrationDetails]:
        """Register object or other types - DRY helper reduces nesting."""
        reg_details = FlextModels.RegistrationDetails(
            registration_id=key,
            handler_mode=FlextConstants.Cqrs.COMMAND_HANDLER_TYPE,
            timestamp="",
            status=FlextConstants.Cqrs.Status.RUNNING,
        )
        return FlextResult[FlextModels.RegistrationDetails].ok(reg_details)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_handler_key(self, handler: FlextHandlers[object, object]) -> str:
        handler_id = getattr(handler, "handler_id", None)
        if isinstance(handler_id, str) and handler_id:
            return handler_id
        return handler.__class__.__name__

    def _resolve_binding_key(
        self,
        handler: FlextHandlers[object, object],
        message_type: type[object],
    ) -> str:
        base_key = self._resolve_handler_key(handler)
        # Handle both type objects and string keys
        if hasattr(message_type, "__name__"):
            type_name = message_type.__name__
        else:
            type_name = str(message_type)
        return f"{base_key}::{type_name}"

    def _resolve_binding_key_from_entry(
        self,
        entry: FlextHandlers[object, object]
        | tuple[
            FlextTypes.Bus.HandlerCallableType,
            object | FlextResult[FlextTypes.GeneralValueType],
        ]
        | object
        | tuple[object, ...]
        | None,
        message_type: type[object],
    ) -> str:
        if isinstance(entry, tuple):
            handler_func = entry[0]
            handler_name = getattr(handler_func, "__name__", str(handler_func))
            # Handle both type objects and string keys
            if hasattr(message_type, "__name__"):
                type_name = message_type.__name__
            else:
                type_name = str(message_type)
            return f"{handler_name}::{type_name}"
        if isinstance(entry, FlextHandlers):
            return self._resolve_binding_key(entry, message_type)
        # Handle object or other types
        return str(entry)

    def register(
        self,
        name: str,
        service: FlextTypes.GeneralValueType,
        metadata: FlextTypes.GeneralValueType | FlextModels.Metadata | None = None,
    ) -> FlextResult[bool]:
        """Register a service component with optional metadata.

        Delegates to the container's register method for dependency injection.
        Metadata is currently stored for future use but not actively used.

        Args:
            name: Service name for later retrieval
            service: Service instance to register
            metadata: Optional metadata (dict or FlextModels.Metadata)

        Returns:
            FlextResult[bool]: Success (True) if registered or failure with error details.

        """
        # Normalize metadata to dict for internal use
        validated_metadata: FlextTypes.GeneralValueType | None = None
        if metadata is not None:
            if FlextRuntime.is_dict_like(metadata):
                validated_metadata = (
                    dict(metadata) if not isinstance(metadata, dict) else metadata
                )
            elif isinstance(metadata, FlextModels.Metadata):
                validated_metadata = metadata.attributes
            else:
                return FlextResult[bool].fail(
                    f"metadata must be dict or FlextModels.Metadata, got {type(metadata).__name__}",
                )

        # Store metadata if provided (for future use)
        if validated_metadata:
            # Log metadata with service name for observability
            self.logger.debug(
                "Registering service with metadata",
                operation="with_service",
                service_name=name,
                has_metadata=True,
                metadata_keys=list(validated_metadata.keys()),
                source="flext-core/src/flext_core/registry.py",
            )

        # Delegate to container
        try:
            self.container.with_service(name, service)
            return FlextResult[bool].ok(True)
        except ValueError as e:
            return FlextResult[bool].fail(str(e))

    # =========================================================================
    # Protocol Implementations: RegistrationTracker, BatchProcessor
    # =========================================================================

    def register_item(
        self,
        name: str,
        item: FlextTypes.GeneralValueType,
    ) -> FlextResult[bool]:
        """Register item (RegistrationTracker protocol)."""
        # Direct implementation without try/except - use FlextResult for error handling
        return self.register(name, item)

    def get_item(self, name: str) -> FlextResult[FlextTypes.GeneralValueType]:
        """Get registered item (RegistrationTracker protocol)."""
        try:
            return self.ok(getattr(self, name))
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return self.fail(str(e))

    def list_items(self) -> FlextResult[list[str]]:
        """List registered items (RegistrationTracker protocol)."""
        try:
            keys = list(getattr(self, "_registered_keys", []))
            return FlextResult[list[str]].ok(keys)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return FlextResult[list[str]].fail(str(e))

    def get_batch_size(self) -> int:
        """Get batch size (BatchProcessor protocol)."""
        return 100


__all__ = ["FlextRegistry"]
