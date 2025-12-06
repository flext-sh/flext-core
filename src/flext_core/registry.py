"""Handler registration and discovery utilities.

FlextRegistry wires handlers to ``FlextDispatcher`` with explicit binding,
idempotent tracking, and batch registration support that matches the current
dispatcher-centric application layer.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, cast

from pydantic import Field, computed_field

from flext_core.constants import c
from flext_core.dispatcher import FlextDispatcher  # For instantiation only
from flext_core.handlers import h
from flext_core.mixins import x
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
    ``p.Application.CommandBus``) for actual handler registration and execution.
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
            ...         m.Handler.RegistrationDetails(
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
            list[m.Handler.RegistrationDetails],
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

    def __init__(self, dispatcher: p.Application.CommandBus | None = None) -> None:
        """Initialize the registry with a CommandBus protocol instance.

        Args:
            dispatcher: CommandBus protocol instance (defaults to creating FlextDispatcher)

        """
        super().__init__()

        # Initialize service infrastructure (DI, Context, Logging, Metrics)
        self._init_service("flext_registry")

        # Structural typing - FlextDispatcher implements p.Application.CommandBus
        # Create dispatcher instance if not provided
        actual_dispatcher: p.Application.CommandBus = (
            dispatcher
            if dispatcher is not None
            else cast("p.Application.CommandBus", FlextDispatcher())
        )
        self._dispatcher: p.Application.CommandBus = actual_dispatcher

        # Enrich context with registry metadata for observability
        self._enrich_context(
            service_type="registry",
            dispatcher_type=type(dispatcher).__name__,
            supports_batch_registration=True,
            idempotent_registration=True,
        )
        self._registered_keys: set[str] = set()

    @staticmethod
    def _safe_get_handler_mode(
        value: t.GeneralValueType,
    ) -> c.Cqrs.HandlerType:
        """Safely extract and validate handler mode from GeneralValueType value."""
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
        """Safely extract and validate status from GeneralValueType value."""
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
        reg_data: t.Types.ConfigurationMapping,
        key: str,
    ) -> m.Handler.RegistrationDetails:
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
        return m.Handler.RegistrationDetails(
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
        handler: (h[t.GeneralValueType, t.GeneralValueType] | None),
    ) -> r[m.Handler.RegistrationDetails]:
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
            return r[m.Handler.RegistrationDetails].fail(
                "Handler cannot be None",
            )

        key = FlextRegistry._resolve_handler_key(handler)
        if key in self._registered_keys:
            self.logger.debug(
                "Handler already registered, returning existing registration",
                operation="register_handler",
                handler_name=handler_name,
                handler_key=key,
                source="flext-core/src/flext_core/registry.py",
            )
            # Return successful registration details for idempotent registration
            return r[m.Handler.RegistrationDetails].ok(
                m.Handler.RegistrationDetails(
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
            source="flext-core/src/flext_core/registry.py",
        )
        # register_handler returns r[t.Types.ConfigurationDict]
        # register_handler accepts GeneralValueType | BaseModel, but h works via runtime check
        # Cast handler to GeneralValueType for type compatibility (runtime handles h correctly)
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
                source="flext-core/src/flext_core/registry.py",
            )
            # Convert GeneralValueType to RegistrationDetails using helper method
            reg_data = registration.value
            reg_details = self._create_registration_details(reg_data, key)
            # Override timestamp with formatted default if not provided
            if not reg_details.timestamp:
                reg_details.timestamp = u.Generators.generate_iso_timestamp().replace(
                    "+00:00",
                    "Z",
                )
            return r[m.Handler.RegistrationDetails].ok(reg_details)

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
        return r[m.Handler.RegistrationDetails].fail(error_msg)

    def register_handlers(
        self,
        handlers: Iterable[h[t.GeneralValueType, t.GeneralValueType]],
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
            result: r[bool] = self._process_single_handler(handler, summary)
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
                return r[FlextRegistry.Summary].fail(error_msg)

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
        handler: h[t.GeneralValueType, t.GeneralValueType],
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
                source="flext-core/src/flext_core/registry.py",
            )
            summary.skipped.append(key)
            return r[bool].ok(True)

        # Handler is already the correct type
        self.logger.debug(
            "Registering handler with dispatcher",
            operation="process_single_handler",
            handler_name=handler_name,
            handler_key=key,
            source="flext-core/src/flext_core/registry.py",
        )
        # register_handler accepts GeneralValueType | BaseModel, but h works via runtime check
        registration_result = self._dispatcher.register_handler(
            cast("t.GeneralValueType", handler),
        )
        if registration_result.is_success:
            # Convert dict result to RegistrationDetails
            reg_data = registration_result.value
            reg_details = self._create_registration_details(reg_data, key)
            if not reg_details:
                # Fallback: create minimal registration details
                reg_details = m.Handler.RegistrationDetails(
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
                source="flext-core/src/flext_core/registry.py",
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
            source="flext-core/src/flext_core/registry.py",
        )
        # Use unwrap_error() for type-safe str
        error_str = registration_result.error or "Unknown error"
        _ = FlextRegistry._add_registration_error(key, error_str, summary)
        return r[FlextRegistry.Summary].fail(error_str)

    def _add_successful_registration(
        self,
        key: str,
        registration: m.Handler.RegistrationDetails,
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
        summary.errors.append(str(error) or f"Failed to register handler '{key}'")
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
                h[t.GeneralValueType, t.GeneralValueType],
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

            # Structural typing - handler is h which implements p.Application.Handler
            # Cast to GeneralValueType for protocol compatibility
            registration = self._dispatcher.register_command(
                message_type,
                cast("t.GeneralValueType", handler),
            )
            if registration.is_failure:
                summary.errors.append(
                    str(registration.error)
                    or f"Failed to register handler '{key}' for '{message_type.__name__}'",
                )
                continue

            self._registered_keys.add(key)
            # Type narrowing: registration.value can be various types, ensure it's dict
            reg_data_raw = registration.value
            if isinstance(reg_data_raw, (dict, Mapping)):
                # Type narrowing: reg_data_raw is Mapping after isinstance check
                reg_data_dict: t.Types.ConfigurationMapping = (
                    reg_data_raw
                    if isinstance(reg_data_raw, dict)
                    else dict(reg_data_raw.items())
                )
                reg_details = self._create_registration_details(
                    reg_data_dict,
                    key,
                )
            else:
                reg_details = m.Handler.RegistrationDetails(
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
                h[t.GeneralValueType, t.GeneralValueType]
                | tuple[
                    t.Handler.HandlerCallable,
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
                        tuple_entry: tuple[
                            t.Handler.HandlerCallable,
                            t.GeneralValueType | r[t.GeneralValueType],
                        ] = (handler_elem, config_elem)
                        result = self._register_tuple_entry(tuple_entry, key)
                    else:
                        result = FlextRegistry._register_other_entry(key)
                elif isinstance(entry, h):
                    # Type narrowing: entry is h instance
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

        return r[FlextRegistry.Summary].ok(summary)

    # ------------------------------------------------------------------
    # Internal helpers for register_function_map (DRY + Reduce nesting)
    # ------------------------------------------------------------------
    def _register_tuple_entry(
        self,
        entry: tuple[
            t.Handler.HandlerCallable,
            t.GeneralValueType | r[t.GeneralValueType],
        ],
        key: str,
    ) -> r[m.Handler.RegistrationDetails]:
        """Register tuple (function, config) entry - DRY helper reduces nesting."""
        handler_func, handler_config = entry

        # Use guard with return_value=True for concise dict validation
        config_dict = (
            u.Validation.guard(handler_config, dict, return_value=True)
            if handler_config is not None
            else None
        )
        # Cast config_dict to HandlerConfigurationType
        handler_config_typed: t.Types.ConfigurationMapping | None = cast(
            "t.Types.ConfigurationMapping | None",
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
            return r[m.Handler.RegistrationDetails].fail(
                f"Failed to create handler: {handler_result.error}",
            )

        # Register with dispatcher
        handler = handler_result.value
        # register_handler accepts GeneralValueType | BaseModel, but h works via runtime check
        register_result = self._dispatcher.register_handler(
            cast("t.GeneralValueType", handler),
        )
        if register_result.is_failure:
            return r[m.Handler.RegistrationDetails].fail(
                f"Failed to register handler: {register_result.error}",
            )

        # Success - create registration details
        reg_details = m.Handler.RegistrationDetails(
            registration_id=key,
            handler_mode=c.Cqrs.HandlerType.COMMAND,
            timestamp="",
            status=c.Cqrs.CommonStatus.RUNNING,
        )
        return r[m.Handler.RegistrationDetails].ok(reg_details)

    def _register_handler_entry(
        self,
        entry: h[t.GeneralValueType, t.GeneralValueType],
        key: str,
    ) -> r[m.Handler.RegistrationDetails]:
        """Register h instance - DRY helper reduces nesting."""
        # register_handler accepts GeneralValueType | BaseModel, but h works via runtime check
        register_result = self._dispatcher.register_handler(
            cast("t.GeneralValueType", entry),
        )
        if register_result.is_failure:
            return r[m.Handler.RegistrationDetails].fail(
                f"Failed to register handler: {register_result.error}",
            )

        # Success - create registration details
        reg_details = m.Handler.RegistrationDetails(
            registration_id=key,
            handler_mode=c.Cqrs.HandlerType.COMMAND,
            timestamp="",
            status=c.Cqrs.CommonStatus.RUNNING,
        )
        return r[m.Handler.RegistrationDetails].ok(reg_details)

    @staticmethod
    def _register_other_entry(
        key: str,
    ) -> r[m.Handler.RegistrationDetails]:
        """Register GeneralValueType or other types - DRY helper reduces nesting."""
        reg_details = m.Handler.RegistrationDetails(
            registration_id=key,
            handler_mode=c.Cqrs.HandlerType.COMMAND,
            timestamp="",
            status=c.Cqrs.CommonStatus.RUNNING,
        )
        return r[m.Handler.RegistrationDetails].ok(reg_details)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_handler_key(
        handler: h[t.GeneralValueType, t.GeneralValueType],
    ) -> str:
        handler_id = getattr(handler, "handler_id", None)
        return (
            handler_id
            if handler_id and u.is_type(handler_id, str)
            else handler.__class__.__name__
        )

    @staticmethod
    def _resolve_binding_key(
        handler: h[t.GeneralValueType, t.GeneralValueType],
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
            h[t.GeneralValueType, t.GeneralValueType]
            | tuple[
                t.Handler.HandlerCallable,
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
        if isinstance(entry, h):
            # Type narrowing: entry is h instance
            return FlextRegistry._resolve_binding_key(entry, message_type)
        # Handle GeneralValueType or other types
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
                # Cast to GeneralValueType for is_dict_like check
                metadata_as_general = cast("t.GeneralValueType", metadata)
                if FlextRuntime.is_dict_like(metadata_as_general):
                    # Type guard ensures metadata_as_general is t.Types.ConfigurationMapping
                    validated_metadata = dict(metadata_as_general.items())
                else:
                    return r[bool].fail(
                        f"metadata must be dict or m.Metadata, got {type(metadata).__name__}",
                    )

        # Store metadata if provided (for future use)
        if validated_metadata and FlextRuntime.is_dict_like(validated_metadata):
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
                source="flext-core/src/flext_core/registry.py",
            )

        # Delegate to container (x.container returns FlextContainer)
        try:
            _ = self.container.with_service(name, service)
            return r[bool].ok(True)
        except ValueError as e:
            return r[bool].fail(str(e))

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
