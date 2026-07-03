"""Infrastructure enforcement catalog rows — extended rows."""

from __future__ import annotations

from typing import Final


class FlextConstantsEnforcementCatalogInfraRowsExtended:
    """Extended infra detector rows extracted for the 200-LOC cap."""

    INFRA_DETECTOR_ROWS_EXTENDED: Final[
        tuple[tuple[str, str, str, str, tuple[str, ...], bool, str], ...]
    ] = (
        (
            "ENFORCE-085",
            "MEDIUM",
            "magic_literal_violations",
            "2-2-facades-namespaces-naming-patterns",
            ("flext-constants-discipline",),
            False,
            "Magic number or string literal used where a named constant exists or should exist.",
        ),
        (
            "ENFORCE-090",
            "HIGH",
            "stub_file_violations",
            "3-4-tools-and-modules",
            ("flext-strict-typing",),
            False,
            "Hand-written `.pyi` stub file detected — source type hints are the SSOT; stubs are prohibited.",
        ),
    )


__all__ = ["FlextConstantsEnforcementCatalogInfraRowsExtended"]
