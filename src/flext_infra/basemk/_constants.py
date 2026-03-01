"""Centralized constants for the basemk subpackage."""

from __future__ import annotations

from typing import Final


class FlextInfraBasemkConstants:
    """Basemk infrastructure constants."""

    TEMPLATE_ORDER: Final[tuple[str, ...]] = (
        "base_header.mk.j2",
        "base_detection.mk.j2",
        "base_venv.mk.j2",
        "base_preflight.mk.j2",
        "base_verbs.mk.j2",
        "base_pr.mk.j2",
        "base_clean.mk.j2",
    )


__all__ = ["FlextInfraBasemkConstants"]
