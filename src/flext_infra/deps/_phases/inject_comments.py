"""Phase: Inject managed/custom/auto markers into pyproject.toml."""

from __future__ import annotations

from flext_infra import c


class InjectCommentsPhase:
    """Inject managed/custom/auto markers into pyproject.toml."""

    def apply(self, rendered: str) -> tuple[str, list[str]]:
        changes: list[str] = []
        lines = rendered.splitlines()
        existing_text = rendered
        out: list[str] = []
        has_banner = bool(
            lines and "[MANAGED] FLEXT pyproject standardization" in lines[0]
        )
        if not has_banner:
            out.extend(c.Infra.Deps.BANNER.splitlines())
            changes.append("managed banner injected")
        marker_map = dict(c.Infra.Deps.COMMENT_MARKERS)
        skip_broken_group_section = False
        for line in lines:
            if line.strip() == "[group.dev.dependencies]":
                skip_broken_group_section = True
                changes.append("broken [group.dev.dependencies] section removed")
                continue
            if skip_broken_group_section and (not line.strip()):
                continue
            if skip_broken_group_section and line.strip():
                skip_broken_group_section = False
            marker = marker_map.get(line.strip())
            if marker:
                recent = (
                    out[-c.Infra.Deps.RECENT_LINES_FOR_MARKER :]
                    if len(out) >= c.Infra.Deps.RECENT_LINES_FOR_MARKER
                    else out
                )
                if marker not in recent and marker not in existing_text:
                    out.append(marker)
                    changes.append(f"marker injected for {line.strip()}")
            if line.strip().startswith("optional-dependencies.dev"):
                recent = (
                    out[-c.Infra.Deps.RECENT_LINES_FOR_DEV_DEP :]
                    if len(out) >= c.Infra.Deps.RECENT_LINES_FOR_DEV_DEP
                    else out
                )
                marker = "# [MANAGED] consolidated development dependencies"
                auto = "# [AUTO] merged from dev/docs/security/test/typings"
                if marker not in recent and marker not in existing_text:
                    out.append(marker)
                    changes.append("marker injected for optional-dependencies.dev")
                if auto not in recent and auto not in existing_text:
                    out.append(auto)
                    changes.append("auto marker injected for optional-dependencies.dev")
            out.append(line)
        return ("\n".join(out).rstrip() + "\n", changes)
