"""Tests for FlextInfraUtilities facade.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextUtilities
from flext_infra import FlextInfraUtilities, u


class TestFlextInfraUtilitiesImport:
    """Test FlextInfraUtilities class import and structure."""

    def test_flext_infra_utilities_is_importable(self) -> None:
        """Test that FlextInfraUtilities can be imported."""
        assert FlextInfraUtilities is not None

    def test_flext_infra_utilities_inherits_from_flext_utilities(self) -> None:
        """Test that FlextInfraUtilities extends FlextUtilities."""
        assert issubclass(FlextInfraUtilities, FlextUtilities)

    def test_runtime_alias_u_is_flext_infra_utilities(self) -> None:
        """Test that u is an alias for FlextInfraUtilities."""
        assert u is FlextInfraUtilities

    def test_flext_infra_utilities_is_class(self) -> None:
        """Test that FlextInfraUtilities is a class."""
        assert isinstance(FlextInfraUtilities, type)

    def test_flext_infra_utilities_can_be_instantiated(self) -> None:
        """Test that FlextInfraUtilities can be instantiated."""
        instance = FlextInfraUtilities()
        assert instance is not None

    def test_flext_infra_utilities_instance_is_flext_infra_utilities(self) -> None:
        """Test that instance is of correct type."""
        instance = FlextInfraUtilities()
        assert isinstance(instance, FlextInfraUtilities)

    def test_flext_infra_utilities_has_inherited_methods(self) -> None:
        """Test that FlextInfraUtilities inherits methods from FlextUtilities."""
        # Check that at least some methods are inherited
        flext_utils_methods = {m for m in dir(FlextUtilities) if not m.startswith("_")}
        infra_utils_methods = {
            m for m in dir(FlextInfraUtilities) if not m.startswith("_")
        }

        # Infra utilities should have at least the same public methods as FlextUtilities
        assert len(infra_utils_methods) >= len(flext_utils_methods)
