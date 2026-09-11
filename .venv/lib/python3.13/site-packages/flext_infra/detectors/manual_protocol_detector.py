"""Detect Protocol classes outside canonical locations via rope.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra import c, m, u

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraManualProtocolDetector:
    """Detect Protocol classes outside canonical protocol files via rope."""

    @staticmethod
    def detect_file(
        ctx: m.Infra.DetectorContext,
    ) -> t.SequenceOf[m.Infra.ManualProtocolViolation]:
        """Detect Protocol classes outside canonical locations."""
        if (
            ctx.file_path.name in c.Infra.FLEXT_PROTOCOLS_FILE_NAMES
            or c.Infra.FLEXT_PROTOCOLS_DIRECTORIES.intersection(ctx.file_path.parts)
        ):
            return []
        if ctx.file_path.name in c.Infra.NAMESPACE_PROTECTED_FILES:
            return []
        file_path = ctx.file_path
        try:
            source = file_path.read_text(encoding=c.Cli.ENCODING_DEFAULT)
        except OSError:
            return []
        return [
            m.Infra.ManualProtocolViolation(
                file=str(file_path), line=ci.line, name=ci.name
            )
            for ci in u.Infra.class_info_from_source(source)
            if "Protocol" in ci.bases
        ]


__all__: list[str] = ["FlextInfraManualProtocolDetector"]
