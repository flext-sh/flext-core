"""Detect loose typing declarations outside canonical typings locations.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra import c, m, u

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraManualTypingAliasDetector:
    """Detect typing declarations outside canonical typings files via rope."""

    @staticmethod
    def detect_file(
        ctx: m.Infra.DetectorContext,
    ) -> t.SequenceOf[m.Infra.ManualTypingAliasViolation]:
        """Detect typing declaration placement violations in a single file."""
        if (
            ctx.file_path.name in c.Infra.FLEXT_TYPINGS_FILE_NAMES
            or c.Infra.FLEXT_TYPINGS_DIRECTORY in ctx.file_path.parts
        ):
            return []
        resource = u.Infra.fetch_python_resource(ctx.rope_project, ctx.file_path)
        if resource is None:
            return []
        file_path = ctx.file_path
        lines = resource.read().splitlines()
        violations: t.MutableSequenceOf[m.Infra.ManualTypingAliasViolation] = []
        for line_number, line in enumerate(lines, start=1):
            if not line or line[0].isspace():
                continue
            pep695_match = c.Infra.PEP695_RE.match(line)
            if pep695_match is not None:
                violations.append(
                    m.Infra.ManualTypingAliasViolation(
                        file=str(file_path),
                        line=line_number,
                        name=pep695_match.group(1),
                        detail="PEP695 alias must be centralized under typings scope",
                    )
                )
                continue
            type_alias_match = c.Infra.TYPEALIAS_ANNOT_RE.match(line)
            if type_alias_match is not None:
                violations.append(
                    m.Infra.ManualTypingAliasViolation(
                        file=str(file_path),
                        line=line_number,
                        name=type_alias_match.group(1),
                        detail="TypeAlias assignment must be centralized under typings scope",
                    )
                )
                continue
            typing_factory_match = c.Infra.TYPING_FACTORY_ASSIGN_RE.match(line)
            if typing_factory_match is None:
                continue
            violations.append(
                m.Infra.ManualTypingAliasViolation(
                    file=str(file_path),
                    line=line_number,
                    name=typing_factory_match.group(1),
                    detail="Typing factory assignment must be centralized under typings scope",
                )
            )
        return violations


__all__: list[str] = ["FlextInfraManualTypingAliasDetector"]
