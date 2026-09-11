"""Hermetic Makefile-only bootstrap for stale generated dispatchers."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from flext_core import r
from flext_infra import c, m
from flext_infra.base import s
from flext_infra.codegen.conform import FlextInfraCodegenConform

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraCodegenMakeBootstrap(s[bool]):
    """Delegate one Makefile projection exclusively to codegen conform."""

    @override
    def execute(self) -> p.Result[bool]:
        """Apply or check only this checkout's canonical Makefile projection."""
        mode = (
            c.Infra.CodegenConformMode.CHECK
            if self.effective_dry_run
            else c.Infra.CodegenConformMode.APPLY
        )
        conformed = FlextInfraCodegenConform.execute_request(
            m.Infra.CodegenConformRequest(
                root=self.repository_root,
                what=c.Infra.CodegenConformSurface.MAKEFILE,
                scope=c.Infra.CodegenConformScope.SELF,
                mode=mode,
            )
        )
        if conformed.failure:
            return r[bool].fail(
                conformed.error or "Makefile bootstrap conformance failed"
            )
        return r[bool].ok(True)


__all__: list[str] = ["FlextInfraCodegenMakeBootstrap"]
