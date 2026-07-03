"""Root package metadata names exposed as top-level lazy attributes."""

from __future__ import annotations

from typing import Final

ROOT_METADATA_NAMES: Final[tuple[str, ...]] = (
    "__author__",
    "__author_email__",
    "__description__",
    "__license__",
    "__title__",
    "__url__",
    "__version__",
    "__version_info__",
)

__all__: list[str] = ["ROOT_METADATA_NAMES"]
