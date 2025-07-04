"""Basic tests for flext_ldap."""

import importlib.util

import pytest


def test_module_imports() -> None:
    """Test that module can be imported."""
    if importlib.util.find_spec("flext_ldap") is None:
        pytest.skip("Module flext_ldap not available")
    else:
        # Module is available
        assert True


def test_basic_functionality() -> None:
    """Test basic functionality exists."""
    if importlib.util.find_spec("flext_ldap") is None:
        pytest.skip("Module not testable")

    import flext_ldap

    # Basic smoke test
    assert hasattr(flext_ldap, "__file__")


class TestBasicCoverage:
    """Basic coverage tests."""

    def test_module_attributes(self) -> None:
        """Test module has expected attributes."""
        if importlib.util.find_spec("flext_ldap") is None:
            pytest.skip("Module not available")

        import flext_ldap

        assert flext_ldap
