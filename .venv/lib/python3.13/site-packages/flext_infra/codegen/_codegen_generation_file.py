"""Canonical generated package artifact selection."""

from __future__ import annotations

from flext_infra import c, m
from flext_infra.codegen._codegen_generation_standard import (
    FlextInfraCodegenGenerationStandardMixin,
)


class FlextInfraCodegenGenerationFileMixin(FlextInfraCodegenGenerationStandardMixin):
    """Render canonical initializer artifacts from one validated plan."""

    @classmethod
    def render_init(cls, plan: m.Infra.LazyInitPlan) -> str:
        """Render a lazy facade for each importable package boundary.

        Real cycle exceptions (bootstrap packages imported during lazy-runtime
        initialization) keep side-effect-free empty inits. All other packages
        get PEP 562 lazy-loading facades.
        """
        segments = frozenset(plan.context.current_pkg.split("."))
        if segments & c.Infra.BOOTSTRAP_CYCLE_EXCEPTION_SEGMENTS:
            return cls._render_static(plan)
        return cls._render_root(plan)


__all__: list[str] = ["FlextInfraCodegenGenerationFileMixin"]
