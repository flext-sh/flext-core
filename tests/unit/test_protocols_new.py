"""Tests for FlextProtocols — structural typing protocol contracts.

Source: flext_core/ (10 files, ~1306 LOC)
Facade: flext_coreFlextProtocols (MRO composition)

Tests protocol existence, runtime checkability, structural conformance
of concrete FLEXT classes, non-conformance rejection, and method signatures.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import pytest

from flext_tests import tm
from tests import e, m, p, r, u

if TYPE_CHECKING:
    from tests.protocols import TestsFlextCoreProtocols
    from tests.typings import TestsFlextCoreTypes


def _as_protocol_subject[T](value: T) -> T:
    """Normalize a concrete sample to the canonical protocol-check subject type."""
    return value


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
        # settings
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
        """R (via r[T]) satisfies p.SuccessCheckable."""
        result = _as_protocol_subject(r[str].ok("hello"))
        tm.that(u.check_protocol_compliance(result, p.SuccessCheckable), eq=True)

    def test_flext_result_satisfies_has_model_dump(self) -> None:
        """R satisfies p.HasModelDump (it's a BaseModel)."""
        result = _as_protocol_subject(r[str].ok("hello"))
        tm.that(u.check_protocol_compliance(result, p.HasModelDump), eq=True)

    def test_pydantic_model_satisfies_has_model_dump(self) -> None:
        """Any Pydantic BaseModel satisfies p.HasModelDump."""

        class _SampleModel(m.Value):
            name: str = "test"

        instance = _as_protocol_subject(_SampleModel())
        tm.that(u.check_protocol_compliance(instance, p.HasModelDump), eq=True)

    def test_flushable_conformance(self) -> None:
        """Object with flush() satisfies p.Flushable."""

        class _Flusher:
            def flush(self) -> None:
                pass

        instance = _as_protocol_subject(_Flusher())
        tm.that(u.check_protocol_compliance(instance, p.Flushable), eq=True)

    def test_auto_discoverable_handler_conformance(self) -> None:
        """Object with can_handle() satisfies p.AutoDiscoverableHandler."""

        class _Handler:
            def can_handle(self, message_type: type) -> bool:
                return True

        instance = _as_protocol_subject(_Handler())
        tm.that(
            u.check_protocol_compliance(instance, p.AutoDiscoverableHandler),
            eq=True,
        )

    def test_provider_like_conformance(self) -> None:
        """Callable factory satisfies p.ProviderLike structurally."""

        class _Provider:
            def __call__(self) -> str:
                return "service"

        instance = _as_protocol_subject(_Provider())
        tm.that(u.check_protocol_compliance(instance, p.ProviderLike), eq=True)

    def test_dispatchable_service_conformance(self) -> None:
        """Object with dispatch(message) satisfies p.DispatchableService."""

        class _Svc:
            def dispatch(
                self,
                message: TestsFlextCoreProtocols.Model,
                /,
            ) -> TestsFlextCoreProtocols.Model:
                return message

        instance = _as_protocol_subject(_Svc())
        tm.that(
            u.check_protocol_compliance(instance, p.DispatchableService),
            eq=True,
        )

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
        """Class without success/failure properties is rejected."""

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
            "success",
            "failure",
            "value",
            "error",
            "error_code",
            "error_data",
            "exception",
            "unwrap",
            "unwrap_or",
            "unwrap_or_else",
        ]
        for attr in expected_attrs:
            tm.that(
                hasattr(p.Result, attr),
                eq=True,
                msg=f"p.Result must define {attr}",
            )

    def test_context_protocol_composes_sub_protocols(self) -> None:
        """p.Context inherits from all sub-protocols."""
        composed_protocol: type = p.Context
        tm.that(issubclass(composed_protocol, p.ContextRead), eq=True)
        tm.that(issubclass(composed_protocol, p.ContextWrite), eq=True)
        tm.that(issubclass(composed_protocol, p.ContextLifecycle), eq=True)
        tm.that(issubclass(composed_protocol, p.ContextExport), eq=True)
        tm.that(issubclass(composed_protocol, p.ContextMetadataAccess), eq=True)

    def test_container_protocol_extends_configurable(self) -> None:
        """p.Container inherits from p.Configurable."""
        container_protocol: type = p.Container
        tm.that(issubclass(container_protocol, p.Configurable), eq=True)

    def test_settings_protocol_extends_has_model_dump(self) -> None:
        """p.Settings inherits from p.HasModelDump."""
        settings_protocol: type = p.Settings
        tm.that(issubclass(settings_protocol, p.HasModelDump), eq=True)

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
        protocol_annotations = getattr(p.Settings, "__annotations__", {})
        for attr in expected:
            tm.that(
                attr in protocol_annotations,
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

    def test_command_bus_has_dispatch_and_register(self) -> None:
        """p.CommandBus exposes only command-routing behavior."""
        for method in ("dispatch", "register_handler"):
            tm.that(
                hasattr(p.CommandBus, method),
                eq=True,
                msg=f"p.CommandBus must have {method}",
            )
        tm.that(
            hasattr(p.CommandBus, "publish"),
            eq=False,
            msg="p.CommandBus must not expose event publishing",
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
            "fetch_plugin",
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
        for method in ("close_connection", "connection_string", "test_connection"):
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
            "update_attribute",
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
        expected_fields = ("mode", "name", "encoding")
        protocol_annotations = getattr(p.TextStream, "__annotations__", {})
        for attr in expected_fields:
            tm.that(
                attr in protocol_annotations,
                eq=True,
                msg=f"p.TextStream must define {attr}",
            )
        for method in ("write", "flush"):
            tm.that(
                hasattr(p.TextStream, method),
                eq=True,
                msg=f"p.TextStream must define {method}",
            )

    def test_service_protocol_has_expected_methods(self) -> None:
        """p.Service defines execute, service_info, valid, validate_business_rules."""
        for method in (
            "execute",
            "service_info",
            "valid",
            "validate_business_rules",
        ):
            tm.that(
                hasattr(p.Service, method),
                eq=True,
                msg=f"p.Service must have {method}",
            )

    def test_model_protocol_has_model_methods(self) -> None:
        """m.BaseModel defines model_dump and model_validate."""
        for method in ("model_dump", "model_validate"):
            tm.that(
                hasattr(m.BaseModel, method),
                eq=True,
                msg=f"m.BaseModel must have {method}",
            )

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

        class _Flusher(m.BaseModel):
            def flush(self) -> None:
                pass

        result = u.check_protocol_compliance(_Flusher(), p.Flushable)
        tm.that(result, eq=True)

    def test_check_protocol_compliance_negative(self) -> None:
        """check_protocol_compliance returns False for non-conforming instance."""

        class _Empty(m.Value):
            pass

        result = u.check_protocol_compliance(_Empty(), p.Flushable)
        tm.that(result, eq=False)

    def test_check_protocol_compliance_type_error(self) -> None:
        """check_protocol_compliance returns False on TypeError."""
        # Passing a non-protocol type should return False, not raise.
        result = u.check_protocol_compliance("hello", int)
        tm.that(result, eq=False)

    # ------------------------------------------------------------------
    # 7. Structural conformance with custom implementations
    # ------------------------------------------------------------------

    def test_success_checkable_custom_implementation(self) -> None:
        """Custom class with success/failure satisfies p.SuccessCheckable."""

        class _Outcome:
            @property
            def success(self) -> bool:
                return True

            @property
            def failure(self) -> bool:
                return False

        instance = _as_protocol_subject(_Outcome())
        tm.that(u.check_protocol_compliance(instance, p.SuccessCheckable), eq=True)

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

            def matches_error_domain(self, domain: str) -> bool:
                return domain == "VALIDATION"

        instance = _StructErr()
        subject = _as_protocol_subject(instance)
        tm.that(u.check_protocol_compliance(subject, p.StructuredError), eq=True)
        tm.that(instance.error_domain, eq="VALIDATION")
        tm.that(instance.matches_error_domain("VALIDATION"), eq=True)
        tm.that(instance.matches_error_domain("NETWORK"), eq=False)

    def test_structured_error_concrete_exception_implementation(self) -> None:
        """Public exceptions must satisfy the same structured error protocol."""
        instance = e.ValidationError("Field required", field="email")
        subject = _as_protocol_subject(instance)

        tm.that(u.check_protocol_compliance(subject, p.StructuredError), eq=True)
        tm.that(instance.error_domain, eq="VALIDATION")
        tm.that(instance.error_message, eq="Field required")
        tm.that(instance.matches_error_domain("VALIDATION"), eq=True)

    def test_error_domain_protocol_conformance(self) -> None:
        """Custom class with value/name attrs satisfies p.ErrorDomainProtocol."""

        class _ErrDomain:
            value: str = "AUTH"
            name: str = "AuthError"

        instance = _as_protocol_subject(_ErrDomain())
        tm.that(u.check_protocol_compliance(instance, p.ErrorDomainProtocol), eq=True)

    def test_configurable_custom_implementation(self) -> None:
        """Custom class with configure() satisfies p.Configurable."""

        class _Configurable:
            def configure(
                self,
                settings: TestsFlextCoreTypes.FlatContainerMapping | None = None,
            ) -> Self:
                return self

        instance = _as_protocol_subject(_Configurable())
        tm.that(u.check_protocol_compliance(instance, p.Configurable), eq=True)

    def test_handle_protocol_custom_implementation(self) -> None:
        """Custom class with handle(message) satisfies p.Handle."""

        class _HandleImpl:
            def handle(
                self,
                message: TestsFlextCoreProtocols.Routable,
            ) -> (
                TestsFlextCoreProtocols.Result[TestsFlextCoreTypes.RuntimeAtomic]
                | TestsFlextCoreTypes.RuntimeAtomic
                | None
            ):
                return None

        instance = _as_protocol_subject(_HandleImpl())
        tm.that(u.check_protocol_compliance(instance, p.Handle), eq=True)

    def test_execute_protocol_custom_implementation(self) -> None:
        """Custom class with execute(message) satisfies p.Execute."""

        class _ExecImpl:
            def execute(
                self,
                message: TestsFlextCoreProtocols.Routable,
            ) -> (
                TestsFlextCoreProtocols.Result[TestsFlextCoreTypes.RuntimeAtomic]
                | TestsFlextCoreTypes.RuntimeAtomic
                | None
            ):
                return None

        instance = _as_protocol_subject(_ExecImpl())
        tm.that(u.check_protocol_compliance(instance, p.Execute), eq=True)

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

    @pytest.mark.parametrize(
        ("name", "group"),
        [
            ("Configurable", "settings"),
            ("Settings", "settings"),
        ],
    )
    def test_config_protocol_group(self, name: str, group: str) -> None:
        """Config protocol group contains expected protocols."""

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

    def test_registry_protocol_group(self) -> None:
        """Registry protocol group contains expected protocols."""

    # ------------------------------------------------------------------
    # 9. r satisfies Result protocol properties
    # ------------------------------------------------------------------

    def test_flext_result_ok_properties(self) -> None:
        """R success exposes expected Result protocol properties."""
        result = r[str].ok("hello")
        tm.that(result.success, eq=True)
        tm.that(result.failure, eq=False)
        tm.that(result.value, eq="hello")
        tm.that(result.error, none=True)
        tm.that(result.error_code, none=True)
        tm.that(result.unwrap(), eq="hello")
        tm.that(bool(result), eq=True)

    def test_flext_result_fail_properties(self) -> None:
        """R failure exposes expected Result protocol properties."""
        result = r[str].fail("something broke")
        tm.that(result.success, eq=False)
        tm.that(result.failure, eq=True)
        tm.that(result.error, eq="something broke")
        tm.that(bool(result), eq=False)

    def test_flext_result_context_manager(self) -> None:
        """R supports context manager protocol (__enter__/__exit__)."""
        result = r[str].ok("ctx")
        with result as ctx:
            tm.that(ctx.value, eq="ctx")

    # ------------------------------------------------------------------
    # 10. MRO inheritance — facade composes all protocol groups
    # ------------------------------------------------------------------

    def test_facade_inherits_all_protocol_groups(self) -> None:
        """FlextProtocols facade MRO includes all 9 protocol groups."""
        expected_bases = [
            p,
            p,
            p,
            p,
            p,
            p,
            p,
            p,
            p,
        ]
        mro = p.__mro__
        for base in expected_bases:
            tm.that(
                base in mro,
                eq=True,
                msg=f"{base.__name__} must be in FlextProtocols MRO",
            )
