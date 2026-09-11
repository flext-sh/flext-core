"""Structural contracts for safe OOXML archive inspection."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol, Self, overload, runtime_checkable


class FlextCliProtocolsXlsxArchive:
    """Narrow contracts for ZIP metadata and defused XML elements."""

    # NOTE (multi-agent, mro-j2yt.1): external objects remain inside the
    # adapter while owned inspection evidence is carried by Pydantic models.
    @runtime_checkable
    class XlsxArchiveInfo(Protocol):
        filename: str
        file_size: int
        compress_size: int

    @runtime_checkable
    class XlsxArchiveReader(Protocol):
        def infolist(
            self,
        ) -> Sequence[FlextCliProtocolsXlsxArchive.XlsxArchiveInfo]: ...

        def read(self, name: str) -> bytes: ...

    @runtime_checkable
    class XlsxXmlElement(Protocol):
        tag: str
        text: str | None

        @overload
        def get(self, key: str, default: None = None) -> str | None: ...

        @overload
        def get[T](self, key: str, default: T) -> str | T: ...

        def iter(self, tag: str | None = None) -> Iterator[Self]: ...


__all__: tuple[str, ...] = ("FlextCliProtocolsXlsxArchive",)
