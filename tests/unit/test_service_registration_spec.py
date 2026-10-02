"""Behavioral tests for the container bootstrap registration spec.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from flext_tests import tm

from flext_core.container import FlextContainer
from tests import e
from tests.constants import c
from tests.models import m

if TYPE_CHECKING:
    from tests.protocols import p


def _factory() -> str:
    return "factory-value"


class TestsServiceRegistrationSpecOwner:
    """The spec validates its declarations; the container registers them."""

    def test_spec_rejects_non_mapping_services(self) -> None:
        """A service collection that is not a mapping fails validation."""
        with pytest.raises(c.ValidationError):
            _ = m.ServiceRegistrationSpec.model_validate({"services": ["invalid"]})

    def test_spec_rejects_non_callable_factory(self) -> None:
        """A factory declaration that is not callable fails validation."""
        with pytest.raises(c.ValidationError):
            _ = m.ServiceRegistrationSpec.model_validate({
                "factories": {"factory": "not-callable"},
            })

    def test_container_registers_the_declared_raw_values(
        self,
        clean_container: p.Container,
    ) -> None:
        """Services, factories and resources declared by a spec all resolve."""
        container = FlextContainer(
            registration=m.ServiceRegistrationSpec(
                services={"service": "value"},
                factories={"factory": _factory},
                resources={"resource": _factory},
            ),
        )

        tm.that(container is clean_container, eq=True)
        tm.that(sorted(container.names()), eq=["factory", "resource", "service"])
        tm.ok(container.resolve("service"), eq="value")
        tm.ok(container.resolve("factory"), eq="factory-value")
        tm.ok(container.resolve("resource"), eq="factory-value")

    def test_spec_rejects_a_prebuilt_record_as_a_factory(self) -> None:
        """A spec declares raw values only; a registration record is not one."""
        record = m.FactoryRegistration(name="factory", factory=_factory)

        with pytest.raises(c.ValidationError):
            _ = m.ServiceRegistrationSpec.model_validate({
                "factories": {"factory": record},
            })

    def test_container_rejects_spec_redeclaring_a_registered_name(
        self,
        clean_container: p.Container,
    ) -> None:
        """Applying a spec to a container that holds its names raises."""
        spec = m.ServiceRegistrationSpec(services={"service": "value"})
        _ = clean_container.bind("service", "first")

        with pytest.raises(e.ValidationError, match="service"):
            _ = FlextContainer(registration=spec)

        tm.ok(clean_container.resolve("service"), eq="first")

    def test_model_declares_no_registration_behavior(self) -> None:
        """The Pydantic model exposes only declarative schema members."""
        behavior_names = {
            "validate_services",
            "validate_factories",
            "validate_resources",
            "_norm_callable_reg",
        }

        tm.that(behavior_names.isdisjoint(m.ServiceRegistrationSpec.__dict__), eq=True)
