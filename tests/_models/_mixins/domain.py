"""Domain and parser model helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from flext_core import m

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.typings import t


class TestsFlextModelsDomainMixin:
    """Domain and parser model helpers."""

    class EmailResponse(m.BaseModel):
        """Shared email response model for tests."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        status: str
        message_id: str

    class NoDict(m.BaseModel):
        """Model for testing value-comparison fallback paths in domain utilities."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=False)

        value: int = 0

        def __init__(self, value: int = 0, **kwargs: t.Scalar) -> None:
            """Initialize model for domain utility edge-case testing."""
            super().__init__(value=value, **kwargs)

        @override
        def __repr__(self) -> str:
            """Return string representation."""
            return f"NoDict(value={self.value})"

    class ParseOptions(m.BaseModel):
        """Test-local parse options after production model removal."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        strip: bool = True
        remove_empty: bool = True
        validator: Callable[[str], bool] | None = None


__all__: list[str] = ["TestsFlextModelsDomainMixin"]
