"""Catalog source models for enforcement.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Literal

from flext_core._constants.enforcement import FlextConstantsEnforcement as ce
from flext_core._typings.base import FlextTypingBase as t

from ._base import (
    EnforcementModelBase,
    FlextModelsEnforcementBase,
)


class FlextModelsEnforcementSources(FlextModelsEnforcementBase):
    """Source-discriminator models used by enforcement catalog rules."""

    class EnforcementInfraDetectorSource(EnforcementModelBase):
        """Rule backed by a ``FlextInfraNamespaceEnforcer`` detector field."""

        kind: Literal["flext_infra_detector"] = "flext_infra_detector"
        violation_field: str
        match_missing: bool = False

    class EnforcementTestsValidatorSource(EnforcementModelBase):
        """Rule backed by a ``FlextTestsValidator`` classmethod."""

        kind: Literal["flext_tests_validator"] = "flext_tests_validator"
        method: str
        rule_ids: t.StrSequence = ()

    class EnforcementRuntimeWarningSource(EnforcementModelBase):
        """Rule backed by a ``warnings`` category raised at runtime."""

        kind: Literal["runtime_warning"] = "runtime_warning"
        category: str

    class EnforcementBeartypeSource(EnforcementModelBase):
        """Rule dispatched through a beartype predicate binding."""

        kind: Literal["beartype"] = "beartype"
        predicate_kind: ce.EnforcementPredicateKind

    class EnforcementRuffSource(EnforcementModelBase):
        """Rule delegated to ruff."""

        kind: Literal["ruff"] = "ruff"
        rule_code: str

    class EnforcementAstGrepSource(EnforcementModelBase):
        """Rule delegated to ast-grep via ``sgconfig.yml``."""

        kind: Literal["ast_grep"] = "ast_grep"
        skill: str
        rule_id: str

    class EnforcementSkillPointerSource(EnforcementModelBase):
        """Rule as narrative skill content only."""

        kind: Literal["skill_pointer"] = "skill_pointer"
        skill: str
        anchor: str = ""


__all__: list[str] = ["FlextModelsEnforcementSources"]
