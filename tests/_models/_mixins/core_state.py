"""Core state model helpers."""

from __future__ import annotations

from typing import ClassVar

from flext_core import m


class TestsFlextModelsCoreStateMixin:
    """Core state model helpers."""

    class SingletonClassForTest(m.BaseModel):
        """Test singleton class with Pydantic validation."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            validate_assignment=True, extra="forbid"
        )

        _instance: ClassVar[
            TestsFlextModelsCoreStateMixin.SingletonClassForTest | None
        ] = None

        name: str = "default"
        timeout: int = 30

        @classmethod
        def fetch_global(cls) -> TestsFlextModelsCoreStateMixin.SingletonClassForTest:
            """Get global singleton instance."""
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

        @classmethod
        def reset_instance(cls) -> None:
            """Reset singleton instance for test isolation."""
            cls._instance = None

    class _SampleEntity(m.BaseModel):
        """Test entity for domain utility tests."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=False)

        unique_id: str = "test-123"
        name: str = "test"


__all__: list[str] = ["TestsFlextModelsCoreStateMixin"]
