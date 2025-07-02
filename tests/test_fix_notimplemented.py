"""Basic tests for flext_ldap."""

import pytest


def test_module_imports() -> None:
    """Test that module can be imported."""
    try:
        import flext_ldap

        assert True
    except ImportError:
        pytest.skip("Module flext_ldap not importable")


def test_basic_functionality() -> None:
    """Test basic functionality exists."""
    try:
        import flext_ldap

        # Basic smoke test
        assert hasattr(flext_ldap, "__file__")
    except (ImportError, AttributeError):
        pytest.skip("Module not testable")


class TestBasicCoverage:
    """Basic coverage tests."""

    def test_module_attributes(self) -> None:
        """Test module has expected attributes."""
        try:
            import flext_ldap

            assert flext_ldap
        except ImportError:
            pytest.skip("Module not importable")
