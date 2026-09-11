"""Shared TYPE_CHECKING stubs for census-rule mixins.

The three census-rules mixins (``_census_rules_dispatch``, ``_census_rules_alias``,
``_census_rules_struct``) each declared identical ``TYPE_CHECKING``-only stubs for
``_detector_context``, ``_raw_violation``, ``_fix_key``, and ``_named_object``
so that the type checker sees the method signatures that are actually resolved
through sibling mixins later in the MRO of ``FlextInfraRefactorCensus``.
This module centralises those stubs in a single base mixin so they are declared once.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import m, p, t


class FlextInfraRefactorCensusRulesSharedMixin:
    """TYPE_CHECKING-only stubs shared by all census-rules mixins.

    At runtime this mixin contributes nothing — the ``if TYPE_CHECKING:``
    block is skipped. The actual method implementations are provided by
    sibling mixins (``_census_objects``, ``_census_apply``, ``_census_symbols``,
    ``_census_filters``) that appear later in the MRO of
    ``FlextInfraRefactorCensus``.
    """

    if TYPE_CHECKING:

        @staticmethod
        def _detector_context(
            rope: p.Infra.RopeWorkspaceDsl,
            file_path: Path,
            *,
            convention: m.Infra.RopeModuleConvention | None = None,
            parse_failures: t.MutableSequenceOf[m.Infra.ParseFailureViolation]
            | None = None,
        ) -> m.Infra.DetectorContext: ...
        @staticmethod
        def _fix_key(file_path: Path, object_name: str, action: str = "") -> str: ...
        @staticmethod
        def _raw_violation(
            *,
            project: str,
            object_name: str,
            object_kind: str,
            kind: str,
            file_path: Path,
            line: int,
            description: str,
            fixable: bool = False,
            fix_action: str = "",
        ) -> m.Infra.Census.Violation: ...
        @staticmethod
        def _named_object(
            objects: tuple[m.Infra.Census.Object, ...], name: str
        ) -> m.Infra.Census.Object | None: ...


__all__: list[str] = ["FlextInfraRefactorCensusRulesSharedMixin"]
