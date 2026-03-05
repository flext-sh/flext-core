"""Method info dataclass for flext_infra.refactor."""

from __future__ import annotations

from dataclasses import dataclass, field

import libcst as cst

from flext_infra.constants import c


def _empty_str_list() -> list[str]:
    return []


@dataclass
class FlextInfraRefactorMethodInfo:
    """Informacoes sobre um metodo para ordenacao."""

    name: str
    category: c.Infra.Refactor.MethodCategory
    node: cst.FunctionDef
    decorators: list[str] = field(default_factory=_empty_str_list)


__all__ = ["FlextInfraRefactorMethodInfo"]
