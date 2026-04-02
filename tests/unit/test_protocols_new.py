"""Tests for FlextProtocols — structural typing protocol contracts.

Source: flext_core._protocols/ (10 files, ~1306 LOC)
Facade: flext_core.protocols.FlextProtocols (MRO composition)

Tests protocol existence, runtime checkability, structural conformance
of concrete FLEXT classes, non-conformance rejection, and method signatures.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Self

import pytest
from pydantic import BaseModel

from flext_core import (
    FlextProtocolsBase,
    FlextProtocolsConfig,
    FlextProtocolsContainer,
    FlextProtocolsContext,
    FlextProtocolsHandler,
    FlextProtocolsLogging,
    FlextProtocolsRegistry,
    FlextProtocolsResult,
    FlextProtocolsService,
    r,
)
from flext_tests import tm
from tests import p, t


class TestFlextProtocols:
    """Tests for all protocol groups accessible via p.* facade."""

    # ------------------------------------------------------------------
    # 1. Protocol existence & accessibility through facade
    # ------------------------------------------------------------------

    _ALL_PROTOCOL_NAMES: list[str] = [
        # base
        "Base",
        "Model",
        "Routable",
        "Executable",
        "Flushable",
        # result
        "Result",
        "HasModelDump",
        "StructuredError",
        "SuccessCheckable",
        "ErrorDomainProtocol",
        # config
        "Configurable",
        "Settings",
        # handler
        "Handler",
        "DispatchMessage",
        "Handle",
        "Execute",
        "Dispatcher",
        "AutoDiscoverableHandler",
        "CommandBus",
        "Middleware",
        # container
        "ProviderLike",
        "ContainerCreationOptions",
        "ContainerCreationOptionsType",
        "Container",
        "RootDict",
        # context
        "ContextRead",
        "ContextWrite",
        "ContextLifecycle",
        "ContextExport",
        "ContextMetadataAccess",
        "Context",
        "RuntimeBootstrapOptions",
        # logging
        "Logger",
        "Metadata",
        "Connection",
        "ValidatorSpec",
        "Entry",
        "TextStream",
        # service
        "CloneableRuntime",
        "Service",
        "DispatchableService",
        # registry
        "Registry",
    ]

    @pytest.mark.parametrize("name", _ALL_PROTOCOL_NAMES)
    def test_protocol_accessible_through_facade(self, name: str) -> None:
        """Every protocol must be accessible via p.<Name>."""
        tm.that(hasattr(p, name), eq=True, msg=f"p.{name} must exist on facade")
        protocol = getattr(p, name)
        tm.that(protocol, none=False, msg=f"p.{name} must not be None")

    @pytest.mark.parametrize("name", _ALL_PROTOCOL_NAMES)
    def test_protocol_is_a_class(self, name: str) -> None:
        """Every protocol must be a class (usable with isinstance/issubclass)."""
        protocol = getattr(p, name)
        tm.that(isinstance(protocol, type), eq=True, msg=f"p.{name} must be a type")

    # ------------------------------------------------------------------
    # 2. Runtime checkability (@runtime_checkable protocols)
    # ------------------------------------------------------------------

    # Protocols with required methods/properties -- empty class MUST be rejected.
    _PROTOCOLS_WITH_REQUIREMENTS: list[str] = [
        "Model",
        "Routable",
        "Executable",
        "Flushable",
        "HasModelDump",
        "StructuredError",
        "SuccessCheckable",
        "Configurable",
        "Handler",
        "DispatchMessage",
        "Handle",
        "Execute",
        "Dispatcher",
        "AutoDiscoverableHandler",
        "CommandBus",
        "Middleware",
        "ProviderLike",
        "ContainerCreationOptionsType",
        "ContextRead",
        "ContextWrite",
        "ContextLifecycle",
        "ContextExport",
        "ContextMetadataAccess",
        "Context",
        "Connection",
        "ValidatorSpec",
        "Entry",
        "Service",
        "DispatchableService",
        "Registry",
    ]

    # Marker/attribute-only protocols -- may match empty classes.
    _MARKER_PROTOCOLS: list[str] = [
        "Base",
        "Result",
        "ErrorDomainProtocol",
        "Settings",
        "ContainerCreationOptions",
        "Container",
        "RuntimeBootstrapOptions",
        "Logger",
        "Metadata",
        "TextStream",
        "CloneableRuntime",
    ]

    @pytest.mark.parametrize("name", _PROTOCOLS_WITH_REQUIREMENTS)
    def test_protocol_rejects_empty_class(self, name: str) -> None:
        """Protocols with required methods reject empty classes via isinstance()."""
        protocol = getattr(p, name)

        class _Empty:
            pass

        try:
            result = isinstance(_Empty(), protocol)
        except TypeError:
            result = False
        tm.that(result, eq=False, msg=f"Empty class must not satisfy p.{name}")

    @pytest.mark.parametrize("name", _MARKER_PROTOCOLS)
    def test_marker_protocol_isinstance_does_not_raise(self, name: str) -> None:
        """Marker/attribute-only protocols support isinstance() without TypeError."""
        protocol = getattr(p, name)

        class _Empty:
            pass

        raised = False
        try:
            isinstance(_Empty(), protocol)
        except TypeError:
            raised = True
        tm.that(raised, eq=False, msg=f"isinstance(x, p.{name}) must not raise")

    # ------------------------------------------------------------------
    # 3. Structural conformance — concrete FLEXT classes satisfy protocols
    # ------------------------------------------------------------------

    def test_flext_result_satisfies_success_checkable(self) -> None:
        """FlextResult (via r[T]) satisfies p.SuccessCheckable."""
        result = r[str].ok("hello")
        tm.that(isinstance(result, p.SuccessCheckable), eq=True)

    def test_flext_result_satisfies_has_model_dump(self) -> None:
        """FlextResult satisfies p.HasModelDump (it's a BaseModel)."""
        result = r[str].ok("hello")
        tm.that(isinstance(result, p.HasModelDump), eq=True)

    def test_pydantic_model_satisfies_has_model_dump(self) -> None:
        """Any Pydantic BaseModel satisfies p.HasModelDump."""

        class _SampleModel(BaseModel):
            name: str = "test"

        instance = _SampleModel()
        tm.that(isinstance(instance, p.HasModelDump), eq=True)

    def test_flushable_conformance(self) -> None:
        """Object with flush() satisfies p.Flushable."""

        class _Flusher:
            def flush(self) -> None:
                pass

        tm.that(isinstance(_Flusher(), p.Flushable), eq=True)

    def test_auto_discoverable_handler_conformance(self) -> None:
        """Object with can_handle() satisfies p.AutoDiscoverableHandler."""

        class _Handler:
            def can_handle(self, message_type: type) -> bool:
                return True

        tm.that(isinstance(_Handler(), p.AutoDiscoverableHandler), eq=True)

    def test_provider_like_conformance(self) -> None:
        """Callable factory satisfies p.ProviderLike structurally."""

        class _Provider:
            def __call__(self) -> str:
                return "service"

        tm.that(isinstance(_Provider(), p.ProviderLike), eq=True)

    def test_dispatchable_service_conformance(self) -> None:
        """Object with dispatch(message) satisfies p.DispatchableService."""

        class _Svc:
            def dispatch(self, message: BaseModel, /) -> BaseModel:
                return message

        tm.that(isinstance(_Svc(), p.DispatchableService), eq=True)

    # ------------------------------------------------------------------
    # 4. Non-conforming classes correctly rejected
    # ------------------------------------------------------------------

    def test_empty_class_rejected_by_has_model_dump(self) -> None:
        """Class without model_dump() does not satisfy p.HasModelDump."""

        class _NoModelDump:
            pass

        tm.that(isinstance(_NoModelDump(), p.HasModelDump), eq=False)

    def test_empty_class_rejected_by_flushable(self) -> None:
        """Class without flush() does not satisfy p.Flushable."""

        class _NoFlush:
            pass

        tm.that(isinstance(_NoFlush(), p.Flushable), eq=False)

    def test_empty_class_rejected_by_auto_discoverable(self) -> None:
        """Class without can_handle() does not satisfy p.AutoDiscoverableHandler."""

        class _NoCanHandle:
            pass

        tm.that(isinstance(_NoCanHandle(), p.AutoDiscoverableHandler), eq=False)

    def test_empty_class_rejected_by_success_checkable(self) -> None:
        """Class without is_success/is_failure properties is rejected."""

        class _NoSuccess:
            pass

        tm.that(isinstance(_NoSuccess(), p.SuccessCheckable), eq=False)

    def test_empty_class_rejected_by_provider_like(self) -> None:
        """Non-callable object is rejected by p.ProviderLike."""

        class _NotCallable:
            pass

        tm.that(isinstance(_NotCallable(), p.ProviderLike), eq=False)

    def test_empty_class_rejected_by_dispatchable_service(self) -> None:
        """Class without dispatch() does not satisfy p.DispatchableService."""

        class _NoDispatch:
            pass

        tm.that(isinstance(_NoDispatch(), p.DispatchableService), eq=False)

    # ------------------------------------------------------------------
    # 5. Protocol method signatures / attributes exist
    # ------------------------------------------------------------------

    def test_result_protocol_has_expected_properties(self) -> None:
        """p.Result defines expected property signatures."""
        expected_attrs = [
            "is_success",
            "is_failure",
            "value",
            "error",
            "error_code",
            "error_data",
            "exception",
            "result_logger",
            "unwrap",
        ]
        for attr in expected_attrs:
            tm.that(
                hasattr(p.Result, attr),
                eq=True,
                msg=f"p.Result must define {attr}",
            )

    def test_context_protocol_composes_sub_protocols(self) -> None:
        """p.Context inherits from all sub-protocols."""
        tm.that(issubclass(p.Context, p.ContextRead), eq=True)
        tm.that(issubclass(p.Context, p.ContextWrite), eq=True)
        tm.that(issubclass(p.Context, p.ContextLifecycle), eq=True)
        tm.that(issubclass(p.Context, p.ContextExport), eq=True)
        tm.that(issubclass(p.Context, p.ContextMetadataAccess), eq=True)

    def test_container_protocol_extends_configurable(self) -> None:
        """p.Container inherits from p.Configurable."""
        tm.that(issubclass(p.Container, p.Configurable), eq=True)

    def test_settings_protocol_extends_has_model_dump(self) -> None:
        """p.Settings inherits from p.HasModelDump."""
        tm.that(issubclass(p.Settings, p.HasModelDump), eq=True)

    def test_settings_protocol_has_expected_attrs(self) -> None:
        """p.Settings defines expected field signatures."""
        expected = [
            "app_name",
            "version",
            "enable_caching",
            "timeout_seconds",
            "dispatcher_auto_context",
            "dispatcher_enable_logging",
        ]
        for attr in expected:
            tm.that(
                attr in p.Settings.__protocol_attrs__
                or attr in getattr(p.Settings, "__annotations__", {}),
                eq=True,
                msg=f"p.Settings must define {attr}",
            )

    def test_dispatcher_extends_message_bus_base(self) -> None:
        """p.Dispatcher has publish and register_handler from _MessageBusBase."""
        for method in ("dispatch", "publish", "register_handler"):
            tm.that(
                hasattr(p.Dispatcher, method),
                eq=True,
                msg=f"p.Dispatcher must have {method}",
            )

    def test_command_bus_has_dispatch_publish_register(self) -> None:
        """p.CommandBus has dispatch, publish, register_handler."""
        for method in ("dispatch", "publish", "register_handler"):
            tm.that(
                hasattr(p.CommandBus, method),
                eq=True,
                msg=f"p.CommandBus must have {method}",
            )

    def test_registry_protocol_has_expected_methods(self) -> None:
        """p.Registry defines handler and plugin management methods."""
        expected = [
            "execute",
            "register",
            "register_handler",
            "register_handlers",
            "register_bindings",
            "register_plugin",
            "unregister_plugin",
            "get_plugin",
            "list_plugins",
        ]
        for method in expected:
            tm.that(
                hasattr(p.Registry, method),
                eq=True,
                msg=f"p.Registry must have {method}",
            )

    def test_connection_protocol_has_expected_methods(self) -> None:
        """p.Connection defines connection lifecycle methods."""
        for method in ("close_connection", "get_connection_string", "test_connection"):
            tm.that(
                hasattr(p.Connection, method),
                eq=True,
                msg=f"p.Connection must have {method}",
            )

    def test_entry_protocol_has_expected_methods(self) -> None:
        """p.Entry defines LDIF entry methods."""
        expected = [
            "attributes",
            "dn",
            "add_attribute",
            "remove_attribute",
            "set_attribute",
            "to_dict",
            "to_ldif",
        ]
        for method in expected:
            tm.that(
                hasattr(p.Entry, method),
                eq=True,
                msg=f"p.Entry must have {method}",
            )

    def test_metadata_protocol_has_expected_properties(self) -> None:
        """p.Metadata defines timestamp and version properties."""
        for attr in ("attributes", "created_at", "updated_at", "version"):
            tm.that(
                hasattr(p.Metadata, attr),
                eq=True,
                msg=f"p.Metadata must have {attr}",
            )

    def test_validator_spec_protocol_has_operators(self) -> None:
        """p.ValidatorSpec defines __call__, __and__, __or__, __invert__."""
        for method in ("__call__", "__and__", "__or__", "__invert__"):
            tm.that(
                hasattr(p.ValidatorSpec, method),
                eq=True,
                msg=f"p.ValidatorSpec must have {method}",
            )

    def test_text_stream_protocol_has_expected_attrs(self) -> None:
        """p.TextStream defines mode, name, encoding, write, flush."""
        for attr in ("mode", "name", "encoding", "write", "flush"):
            tm.that(
                attr in getattr(p.TextStream, "__protocol_attrs__", set())
                or attr in getattr(p.TextStream, "__annotations__", {}),
                eq=True,
                msg=f"p.TextStream must define {attr}",
            )

    def test_service_protocol_has_expected_methods(self) -> None:
        """p.Service defines execute, get_service_info, is_valid, validate_business_rules."""
        for method in (
            "execute",
            "get_service_info",
            "is_valid",
            "validate_business_rules",
        ):
            tm.that(
                hasattr(p.Service, method),
                eq=True,
                msg=f"p.Service must have {method}",
            )

    def test_model_protocol_has_model_methods(self) -> None:
        """p.Model defines model_dump and model_validate."""
        tm.that(hasattr(p.Model, "model_dump"), eq=True)
        tm.that(hasattr(p.Model, "model_validate"), eq=True)

    def test_routable_protocol_has_type_properties(self) -> None:
        """p.Routable defines command_type, event_type, query_type."""
        for attr in ("command_type", "event_type", "query_type"):
            tm.that(
                hasattr(p.Routable, attr),
                eq=True,
                msg=f"p.Routable must define {attr}",
            )

    # ------------------------------------------------------------------
    # 6. Protocol utility methods (check/validate compliance)
    # ------------------------------------------------------------------

    def test_check_protocol_compliance_positive(self) -> None:
        """check_protocol_compliance returns True for conforming instance."""

        class _Flusher:
            def flush(self) -> None:
                pass

        result = p.check_protocol_compliance(_Flusher(), p.Flushable)
        tm.that(result, eq=True)

    def test_check_protocol_compliance_negative(self) -> None:
        """check_protocol_compliance returns False for non-conforming instance."""

        class _Empty:
            pass

        result = p.check_protocol_compliance(_Empty(), p.Flushable)
        tm.that(result, eq=False)

    def test_check_protocol_compliance_type_error(self) -> None:
        """check_protocol_compliance returns False on TypeError."""
        # Passing a non-protocol type should return False, not raise.
        result = p.check_protocol_compliance("hello", int)
        tm.that(result, eq=False)

    def test_validate_protocol_compliance_passes(self) -> None:
        """validate_protocol_compliance does not raise for conforming class."""

        class _Flusher:
            def flush(self) -> None:
                pass

        # Should not raise
        p.validate_protocol_compliance(_Flusher, p.Flushable, "_Flusher")

    def test_validate_protocol_compliance_raises(self) -> None:
        """validate_protocol_compliance raises TypeError for non-conforming class."""

        class _Empty:
            pass

        with pytest.raises(TypeError, match="does not implement protocol"):
            p.validate_protocol_compliance(_Empty, p.Flushable, "_Empty")

    # ------------------------------------------------------------------
    # 7. Structural conformance with custom implementations
    # ------------------------------------------------------------------

    def test_success_checkable_custom_implementation(self) -> None:
        """Custom class with is_success/is_failure satisfies p.SuccessCheckable."""

        class _Outcome:
            @property
            def is_success(self) -> bool:
                return True

            @property
            def is_failure(self) -> bool:
                return False

        tm.that(isinstance(_Outcome(), p.SuccessCheckable), eq=True)

    def test_structured_error_custom_implementation(self) -> None:
        """Custom class satisfying p.StructuredError contract."""

        class _StructErr:
            @property
            def error_domain(self) -> str | None:
                return "VALIDATION"

            @property
            def error_code(self) -> str | None:
                return "FIELD_REQUIRED"

            @property
            def error_message(self) -> str | None:
                return "Field X is required"

            def is_error_domain(self, domain: str) -> bool:
                return domain == "VALIDATION"

        instance = _StructErr()
        tm.that(isinstance(instance, p.StructuredError), eq=True)
        tm.that(instance.error_domain, eq="VALIDATION")
        tm.that(instance.is_error_domain("VALIDATION"), eq=True)
        tm.that(instance.is_error_domain("NETWORK"), eq=False)

    def test_error_domain_protocol_conformance(self) -> None:
        """Custom class with value/name attrs satisfies p.ErrorDomainProtocol."""

        class _ErrDomain:
            value: str = "AUTH"
            name: str = "AuthError"

        tm.that(isinstance(_ErrDomain(), p.ErrorDomainProtocol), eq=True)

    def test_configurable_custom_implementation(self) -> None:
        """Custom class with configure() satisfies p.Configurable."""

        class _Configurable:
            def configure(
                self,
                config: t.FlatContainerMapping | None = None,
            ) -> Self:
                return self

        tm.that(isinstance(_Configurable(), p.Configurable), eq=True)

    def test_handle_protocol_custom_implementation(self) -> None:
        """Custom class with handle(message) satisfies p.Handle."""

        class _HandleImpl:
            def handle(
                self,
                message: p.Routable,
            ) -> r[t.RuntimeAtomic] | t.Container | BaseModel | None:
                return None

        tm.that(isinstance(_HandleImpl(), p.Handle), eq=True)

    def test_execute_protocol_custom_implementation(self) -> None:
        """Custom class with execute(message) satisfies p.Execute."""

        class _ExecImpl:
            def execute(
                self,
                message: p.Routable,
            ) -> r[t.RuntimeAtomic] | t.Container | BaseModel | None:
                return None

        tm.that(isinstance(_ExecImpl(), p.Execute), eq=True)

    # ------------------------------------------------------------------
    # 8. Protocol group completeness — base protocols
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("name", "group"),
        [
            ("Base", "base"),
            ("Model", "base"),
            ("Routable", "base"),
            ("Executable", "base"),
            ("Flushable", "base"),
        ],
    )
    def test_base_protocol_group(self, name: str, group: str) -> None:
        """Base protocol group contains expected protocols."""
        tm.that(hasattr(p, name), eq=True, msg=f"p.{name} from {group} group")

    @pytest.mark.parametrize(
        ("name", "group"),
        [
            ("Result", "result"),
            ("HasModelDump", "result"),
            ("StructuredError", "result"),
            ("SuccessCheckable", "result"),
            ("ErrorDomainProtocol", "result"),
        ],
    )
    def test_result_protocol_group(self, name: str, group: str) -> None:
        """Result protocol group contains expected protocols."""
        tm.that(hasattr(p, name), eq=True, msg=f"p.{name} from {group} group")

    @pytest.mark.parametrize(
        ("name", "group"),
        [
            ("Configurable", "config"),
            ("Settings", "config"),
        ],
    )
    def test_config_protocol_group(self, name: str, group: str) -> None:
        """Config protocol group contains expected protocols."""
        tm.that(hasattr(p, name), eq=True, msg=f"p.{name} from {group} group")

    @pytest.mark.parametrize(
        ("name", "group"),
        [
            ("Handler", "handler"),
            ("DispatchMessage", "handler"),
            ("Handle", "handler"),
            ("Execute", "handler"),
            ("Dispatcher", "handler"),
            ("AutoDiscoverableHandler", "handler"),
            ("CommandBus", "handler"),
            ("Middleware", "handler"),
        ],
    )
    def test_handler_protocol_group(self, name: str, group: str) -> None:
        """Handler protocol group contains expected protocols."""
        tm.that(hasattr(p, name), eq=True, msg=f"p.{name} from {group} group")

    @pytest.mark.parametrize(
        ("name", "group"),
        [
            ("ProviderLike", "container"),
            ("ContainerCreationOptions", "container"),
            ("ContainerCreationOptionsType", "container"),
            ("Container", "container"),
            ("RootDict", "container"),
        ],
    )
    def test_container_protocol_group(self, name: str, group: str) -> None:
        """Container protocol group contains expected protocols."""
        tm.that(hasattr(p, name), eq=True, msg=f"p.{name} from {group} group")

    @pytest.mark.parametrize(
        ("name", "group"),
        [
            ("ContextRead", "context"),
            ("ContextWrite", "context"),
            ("ContextLifecycle", "context"),
            ("ContextExport", "context"),
            ("ContextMetadataAccess", "context"),
            ("Context", "context"),
            ("RuntimeBootstrapOptions", "context"),
        ],
    )
    def test_context_protocol_group(self, name: str, group: str) -> None:
        """Context protocol group contains expected protocols."""
        tm.that(hasattr(p, name), eq=True, msg=f"p.{name} from {group} group")

    @pytest.mark.parametrize(
        ("name", "group"),
        [
            ("Logger", "logging"),
            ("Metadata", "logging"),
            ("Connection", "logging"),
            ("ValidatorSpec", "logging"),
            ("Entry", "logging"),
            ("TextStream", "logging"),
        ],
    )
    def test_logging_protocol_group(self, name: str, group: str) -> None:
        """Logging protocol group contains expected protocols."""
        tm.that(hasattr(p, name), eq=True, msg=f"p.{name} from {group} group")

    @pytest.mark.parametrize(
        ("name", "group"),
        [
            ("CloneableRuntime", "service"),
            ("Service", "service"),
            ("DispatchableService", "service"),
        ],
    )
    def test_service_protocol_group(self, name: str, group: str) -> None:
        """Service protocol group contains expected protocols."""
        tm.that(hasattr(p, name), eq=True, msg=f"p.{name} from {group} group")

    def test_registry_protocol_group(self) -> None:
        """Registry protocol group contains expected protocols."""
        tm.that(hasattr(p, "Registry"), eq=True, msg="p.Registry from registry group")

    # ------------------------------------------------------------------
    # 9. FlextResult satisfies Result protocol properties
    # ------------------------------------------------------------------

    def test_flext_result_ok_properties(self) -> None:
        """FlextResult success exposes expected Result protocol properties."""
        result = r[str].ok("hello")
        tm.that(result.is_success, eq=True)
        tm.that(result.is_failure, eq=False)
        tm.that(result.value, eq="hello")
        tm.that(result.error, none=True)
        tm.that(result.error_code, none=True)
        tm.that(result.unwrap(), eq="hello")
        tm.that(bool(result), eq=True)

    def test_flext_result_fail_properties(self) -> None:
        """FlextResult failure exposes expected Result protocol properties."""
        result = r[str].fail("something broke")
        tm.that(result.is_success, eq=False)
        tm.that(result.is_failure, eq=True)
        tm.that(result.error, eq="something broke")
        tm.that(bool(result), eq=False)

    def test_flext_result_context_manager(self) -> None:
        """FlextResult supports context manager protocol (__enter__/__exit__)."""
        result = r[str].ok("ctx")
        with result as ctx:
            tm.that(ctx.value, eq="ctx")

    # ------------------------------------------------------------------
    # 10. MRO inheritance — facade composes all protocol groups
    # ------------------------------------------------------------------

    def test_facade_inherits_all_protocol_groups(self) -> None:
        """FlextProtocols facade MRO includes all 9 protocol groups."""
        expected_bases = [
            FlextProtocolsBase,
            FlextProtocolsConfig,
            FlextProtocolsContainer,
            FlextProtocolsContext,
            FlextProtocolsHandler,
            FlextProtocolsLogging,
            FlextProtocolsRegistry,
            FlextProtocolsResult,
            FlextProtocolsService,
        ]
        mro = p.__mro__
        for base in expected_bases:
            tm.that(
                base in mro,
                eq=True,
                msg=f"{base.__name__} must be in FlextProtocols MRO",
            )
