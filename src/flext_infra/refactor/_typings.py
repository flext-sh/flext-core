"""Type aliases for refactor-domain policy wiring."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

from flext_infra import m


class FlextInfraRectorTypes:
    PolicyContext: TypeAlias = Mapping[str, m.Infra.Refactor.ClassNestingPolicy]
    ClassFamilyMap: TypeAlias = Mapping[str, str]


__all__ = ["FlextInfraRectorTypes"]
