"""Tests for constants_new/__init__.py module.

Tests the facade package that exposes only FlextConstants.
"""

from flext_core.constants_new import FlextConstants


class TestConstantsNewInit:
    """Tests for constants_new init module facade."""

    def test_import_flext_constants(self) -> None:
        """Test that FlextConstants can be imported from the package."""
        assert FlextConstants is not None
        assert hasattr(FlextConstants, "__name__")

    def test_all_exports(self) -> None:
        """Test that __all__ contains expected exports."""
        from flext_core.constants_new import __all__

        assert "__all__" in dir(__import__("flext_core.constants_new"))
        assert "FlextConstants" in __all__
        assert len(__all__) == 1

    def test_flext_constants_class_exists(self) -> None:
        """Test that FlextConstants class is properly imported."""
        assert hasattr(FlextConstants, "__class__")
        assert FlextConstants.__name__ == "FlextConstants"

    def test_package_imports(self) -> None:
        """Test importing the full package structure."""
        from flext_core import constants_new

        assert hasattr(constants_new, "FlextConstants")
        assert constants_new.FlextConstants is FlextConstants

    def test_direct_import_from_package(self) -> None:
        """Test direct import functionality."""
        from flext_core.constants_new import FlextConstants as ImportedConstants

        assert ImportedConstants is FlextConstants
        assert ImportedConstants.__name__ == "FlextConstants"
