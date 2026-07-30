"""Characterization tests for container bootstrap registration parsing."""

from __future__ import annotations

from typing import cast

import pytest

from flext_core import u
from flext_tests import tm
from tests.models import m


def _factory() -> str:
    return "factory-value"


class TestsServiceRegistrationSpecOwner:
    """Behavioral ownership contract for bootstrap registration normalization."""

    def test_utility_normalizes_raw_registration_mappings(self) -> None:
        """The canonical utility converts every raw registration mapping."""
        registration = u.normalize_service_registration_spec(
            m.ServiceRegistrationSpec.model_construct(
                services={"service": "value"},
                factories={"factory": _factory},
                resources={"resource": _factory},
            )
        )

        tm.that(registration.services is not None, eq=True)
        services = registration.services or {}

        service_record = cast("m.ServiceRegistration", services["service"])
        assert isinstance(service_record, m.ServiceRegistration)
        tm.that(service_record.name, eq="service")
        tm.that(service_record.service, eq="value")
        tm.that(service_record.service_type, eq="str")
        tm.that(registration.factories is not None, eq=True)
        factories = registration.factories or {}
        factory_record = cast("m.FactoryRegistration", factories["factory"])
        assert isinstance(factory_record, m.FactoryRegistration)
        tm.that(factory_record.name, eq="factory")
        tm.that(factory_record.factory is _factory, eq=True)
        tm.that(registration.resources is not None, eq=True)
        resources = registration.resources or {}
        resource_record = cast("m.ResourceRegistration", resources["resource"])
        assert isinstance(resource_record, m.ResourceRegistration)
        tm.that(resource_record.name, eq="resource")
        tm.that(resource_record.factory is _factory, eq=True)

    def test_utility_preserves_non_mapping_services_error(self) -> None:
        """Malformed service collections retain the characterized error contract."""
        registration = m.ServiceRegistrationSpec.model_construct(services=["invalid"])

        with pytest.raises(AttributeError, match="has no attribute 'items'"):
            _ = u.normalize_service_registration_spec(registration)

    def test_utility_preserves_prebuilt_registration_records(self) -> None:
        """Already-normalized registrations retain their object identity."""
        service = m.ServiceRegistration(
            name="service", service="value", service_type="str"
        )
        factory = m.FactoryRegistration(name="factory", factory=_factory)
        resource = m.ResourceRegistration(name="resource", factory=_factory)

        registration = u.normalize_service_registration_spec(
            m.ServiceRegistrationSpec(
                services={"service": service},
                factories={"factory": factory},
                resources={"resource": resource},
            )
        )

        tm.that((registration.services or {})["service"] is service, eq=True)
        tm.that((registration.factories or {})["factory"] is factory, eq=True)
        tm.that((registration.resources or {})["resource"] is resource, eq=True)

    def test_model_declares_no_registration_behavior(self) -> None:
        """The Pydantic model exposes only declarative schema members."""
        behavior_names = {
            "validate_services",
            "validate_factories",
            "validate_resources",
            "_norm_callable_reg",
        }

        tm.that(behavior_names.isdisjoint(m.ServiceRegistrationSpec.__dict__), eq=True)
