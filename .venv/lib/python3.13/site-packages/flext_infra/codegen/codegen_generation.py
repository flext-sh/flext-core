"""Lazy-init file generation facade."""

from __future__ import annotations

from flext_infra.codegen._codegen_generation_file import (
    FlextInfraCodegenGenerationFileMixin,
)


class FlextInfraCodegenGeneration(FlextInfraCodegenGenerationFileMixin):
    """Generate Python module files with lazy import infrastructure."""


__all__: list[str] = ["FlextInfraCodegenGeneration"]
