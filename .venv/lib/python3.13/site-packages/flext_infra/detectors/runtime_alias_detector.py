"""Detect missing or duplicate runtime alias assignments via rope.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra import c, m, u

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraRuntimeAliasDetector:
    """Detect missing/duplicate runtime aliases (e.g. m = FlextFooModels) via rope."""

    @staticmethod
    def detect_file(
        ctx: m.Infra.DetectorContext,
    ) -> t.SequenceOf[m.Infra.RuntimeAliasViolation]:
        """Detect missing/duplicate runtime alias assignments in a facade file."""
        file_path = ctx.file_path
        family = c.Infra.NAMESPACE_FILE_TO_FAMILY.get(file_path.name)
        if family is None:
            return []
        parts = file_path.parts
        if c.Infra.RUNTIME_ALIAS_PARTS_SKIP & frozenset(parts):
            return []
        if ctx.project_root is not None:
            try:
                rel = file_path.relative_to(ctx.project_root)
            except ValueError:
                rel = None
            if rel is not None:
                rel_parts = rel.parts
                if (
                    len(rel_parts) >= c.Infra.RUNTIME_ALIAS_SRC_DEPTH_MIN
                    and rel_parts[0] == "src"
                ):
                    # src/<package>/<facade>.py only; nested submodules are not root facades.
                    if len(rel_parts) != c.Infra.RUNTIME_ALIAS_SRC_DEPTH_EXACT:
                        return []
                elif rel_parts and rel_parts[0] in c.Infra.RUNTIME_ALIAS_NON_ROOT_DIRS:
                    # tests/<file>.py or examples/<file>.py or scripts/<file>.py only.
                    if len(rel_parts) != c.Infra.RUNTIME_ALIAS_NON_ROOT_DEPTH_EXACT:
                        return []
                else:
                    return []
        resource = u.Infra.fetch_python_resource(
            ctx.rope_project, file_path, skip_protected=True
        )
        if resource is None:
            return []
        source = resource.read()
        matches = [
            hit.group(2)
            for hit in c.Infra.FACADE_ALIAS_RE.finditer(source)
            if hit.group(1) == family
        ]
        if not matches:
            return [
                m.Infra.RuntimeAliasViolation(
                    file=str(file_path),
                    kind="missing",
                    alias=family,
                    detail=f"No '{family} = ...' assignment found",
                )
            ]
        if len(matches) > 1:
            return [
                m.Infra.RuntimeAliasViolation(
                    file=str(file_path),
                    kind="duplicate",
                    alias=family,
                    detail=f"Found {len(matches)} '{family} = ...' assignments",
                )
            ]
        return []


__all__: list[str] = ["FlextInfraRuntimeAliasDetector"]
