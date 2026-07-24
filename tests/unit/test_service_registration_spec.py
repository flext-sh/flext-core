"""Characterization tests for container bootstrap registration parsing."""

from __future__ import annotations

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
        tm.that(services["service"].name, eq="service")
        tm.that(services["service"].service, eq="value")
        tm.that(services["service"].service_type, eq="str")
        tm.that(registration.factories is not None, eq=True)
        factories = registration.factories or {}
        tm.that(factories["factory"].name, eq="factory")
        tm.that(factories["factory"].factory is _factory, eq=True)
        tm.that(registration.resources is not None, eq=True)
        resources = registration.resources or {}
        tm.that(resources["resource"].name, eq="resource")
        tm.that(resources["resource"].factory is _factory, eq=True)

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
