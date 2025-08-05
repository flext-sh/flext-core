"""Coverage boost tests for mixins module - target 95% coverage."""

from __future__ import annotations

from flext_core.mixins import FlextFullMixin


class TestFullMixin:
    """Test full mixin methods that need coverage."""

    def test_configure_enterprise_features_with_id(self) -> None:
        """Test configure_enterprise_features with ID parameter."""

        class TestEntity(FlextFullMixin):
            """Test entity with enterprise mixin."""

            def __init__(self) -> None:
                super().__init__()
                self._id: str | None = None

            def set_id(self, entity_id: str) -> None:
                """Set entity ID."""
                self._id = entity_id

            @property
            def id(self) -> str | None:
                """Get entity ID."""
                return self._id

        entity = TestEntity()

        # Test with ID parameter - this should cover line 274-275
        entity.configure_enterprise_features(id="test_id", entity_name="TestName")

        # Verify ID was set
        assert entity.id == "test_id"

    def test_configure_enterprise_features_without_id(self) -> None:
        """Test configure_enterprise_features without ID parameter."""

        class TestEntity(FlextFullMixin):
            """Test entity with enterprise mixin."""

            def __init__(self) -> None:
                super().__init__()
                self._id: str | None = None

            def set_id(self, entity_id: str) -> None:
                """Set entity ID."""
                self._id = entity_id

            @property
            def id(self) -> str | None:
                """Get entity ID."""
                return self._id

        entity = TestEntity()

        # Test without ID parameter - this should skip the ID setting block
        entity.configure_enterprise_features(entity_name="TestName")

        # Verify ID was not set
        assert entity.id is None
