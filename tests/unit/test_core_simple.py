"""Simple test suite for FlextCore - tests only existing methods."""

from __future__ import annotations

from flext_core import FlextCore


class TestFlextCoreSimple:
    """Simple test suite for FlextCore to test only existing methods."""

    def test_basic_functionality(self) -> None:
        """Test basic FlextCore functionality."""
        core = FlextCore()

        # Test basic properties
        assert isinstance(core.entity_id, str)
        assert len(core.entity_id) > 0

        # Test session ID
        session_id = core.get_session_id()
        assert isinstance(session_id, str)
        assert session_id.startswith("session_")

        # Test container access
        container = core.container
        assert container is not None

        # Test cleanup
        cleanup_result = core.cleanup()
        assert cleanup_result.is_success

        # Test string representations
        str_repr = str(core)
        assert "FlextCore" in str_repr
        assert core.entity_id in str_repr

        repr_str = repr(core)
        assert "FlextCore" in repr_str
        assert core.entity_id in repr_str

    def test_singleton_functionality(self) -> None:
        """Test singleton functionality."""
        # Test get_instance
        instance1 = FlextCore.get_instance()
        instance2 = FlextCore.get_instance()
        assert instance1 is instance2

        # Test reset_instance
        FlextCore.reset_instance()
        instance3 = FlextCore.get_instance()
        assert instance3 is not instance1

    def test_direct_access_properties(self) -> None:
        """Test direct access to flext-core components."""
        core = FlextCore()

        # Test that all actual properties exist - based on real API
        assert hasattr(core, "cleanup")
        assert hasattr(core, "container")
        assert hasattr(core, "entity_id")
        assert hasattr(core, "get_config")
        assert hasattr(core, "get_instance")
        assert hasattr(core, "get_session_id")
        assert hasattr(core, "reset_instance")

        # Interface atual do FlextCore - verifica propriedades disponíveis
        assert hasattr(core, "Config")
        assert hasattr(core, "Models")
        assert hasattr(core, "Commandsds")  # nome atual no core.py
        assert hasattr(core, "Result")
        assert hasattr(core, "Container")  # nome atual no core.py
        assert hasattr(core, "Context")
        assert hasattr(core, "Logger")
        assert hasattr(core, "Constants")

    def test_multiple_instances(self) -> None:
        """Test that multiple instances work correctly."""
        core1 = FlextCore()
        core2 = FlextCore()

        # Each instance should have unique entity_id
        assert core1.entity_id != core2.entity_id

        # Each instance should have unique session_id
        assert core1.get_session_id() != core2.get_session_id()

        # But they should share the same container
        assert core1.container is core2.container
