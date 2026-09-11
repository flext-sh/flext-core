"""Phase: Inject managed/custom markers into pyproject.toml."""

from __future__ import annotations

from flext_infra import c, config, t


class FlextInfraInjectCommentsPhase:
    """Inject managed/custom markers into pyproject.toml."""

    _MYPY_RATIONALE_HEADER = (
        "# FLEXT mypy suppression rationale (validated at the facade-FLEXT boundary):"
    )
    _RUFF_RATIONALE_HEADER = (
        "# FLEXT Ruff suppression rationale (validated against semantic facet order):"
    )
    _PYRIGHT_RATIONALE_HEADER = "# FLEXT Pyright suppression rationale (validated at the facade-FLEXT boundary):"

    @classmethod
    def _mypy_rationale_lines(cls) -> t.StrSequence:
        """Render evidence-backed Mypy exclusions from the tooling SSOT."""
        return (
            cls._MYPY_RATIONALE_HEADER,
            *(
                f"# FLEXT mypy[{code}]: {rationale}"
                for code, rationale in sorted(
                    config.Infra.tooling.tools.mypy.disabled_error_codes.items()
                )
            ),
        )

    @classmethod
    def _ruff_rationale_lines(cls) -> t.StrSequence:
        """Render evidence-backed Ruff exclusions from the tooling SSOT."""
        return (
            cls._RUFF_RATIONALE_HEADER,
            *(
                f"# FLEXT ruff[{code}]: {rationale}"
                for code, rationale in sorted(
                    config.Infra.tooling.tools.ruff.lint.ignored_rule_rationales.items()
                )
            ),
        )

    @classmethod
    def _pyright_rationale_lines(cls) -> t.StrSequence:
        """Render evidence-backed Pyright exclusions from the tooling SSOT."""
        return (
            cls._PYRIGHT_RATIONALE_HEADER,
            *(
                f"# FLEXT pyright[{code}]: {rationale}"
                for code, rationale in sorted(
                    config.Infra.tooling.tools.pyright.global_suppression_rationales.items()
                )
            ),
        )

    @staticmethod
    def _is_section_header(line: str) -> bool:
        """Is section header."""
        stripped = line.strip()
        return stripped.startswith("[") and stripped.endswith("]")

    @classmethod
    def _managed_marker_lines(cls) -> t.Infra.StrSet:
        """Return the managed marker lines."""
        markers = {marker for _section_prefix, marker in c.Infra.COMMENT_MARKERS}
        markers.add(c.Infra.DEV_OPTIONAL_DEPS_MARKER)
        markers.add(c.Infra.LEGACY_AUTO_MARKER)
        markers.add(c.Infra.LEGACY_AUTO_BANNER_LINE)
        markers.update(c.Infra.BANNER.splitlines())
        markers.update(cls._mypy_rationale_lines())
        markers.update(cls._ruff_rationale_lines())
        markers.update(cls._pyright_rationale_lines())
        return markers

    @staticmethod
    def _marker_for_section(section_header: str) -> str | None:
        """Marker for section."""
        for section_prefix, marker_text in c.Infra.COMMENT_MARKERS:
            if section_header.startswith(section_prefix):
                marker: str = marker_text
                return marker
        return None

    @classmethod
    def _strip_managed_lines(
        cls, lines: t.StrSequence
    ) -> t.Pair[t.StrSequence, t.StrSequence]:
        """Strip managed lines."""
        changes: t.MutableSequenceOf[str] = []
        managed_lines = cls._managed_marker_lines()
        cleaned: t.MutableSequenceOf[str] = []
        skip_broken_group_section = False
        broken_removed = False
        for line in lines:
            stripped = line.strip()
            if skip_broken_group_section:
                if cls._is_section_header(line):
                    skip_broken_group_section = False
                else:
                    continue
            # Remove legacy banner variants so canonical banner can be re-injected.
            if stripped.startswith("# Sections with [MANAGED] are enforced"):
                continue
            if stripped == c.Infra.LEGACY_AUTO_BANNER_LINE:
                continue
            if stripped.startswith("# FLEXT mypy["):
                continue
            if stripped.startswith("# FLEXT ruff["):
                continue
            if stripped.startswith("# FLEXT pyright["):
                continue
            if stripped == "[group.dev.dependencies]":
                skip_broken_group_section = True
                broken_removed = True
                continue
            if stripped in managed_lines:
                continue
            cleaned.append(line)
        if broken_removed:
            changes.append("broken [group.dev.dependencies] section removed")
        return cleaned, changes

    @staticmethod
    def _inject_dev_markers(
        out: t.MutableSequenceOf[str],
        changes: t.MutableSequenceOf[str],
        emitted_markers: t.Infra.StrSet,
    ) -> None:
        """Inject dev markers."""
        managed_marker = c.Infra.DEV_OPTIONAL_DEPS_MARKER
        if managed_marker not in emitted_markers:
            out.append(managed_marker)
            changes.append("marker injected for optional-dependencies.dev")
            emitted_markers.add(managed_marker)

    @staticmethod
    def _collapse_blank_lines(lines: t.StrSequence) -> t.StrSequence:
        """Collapse repeated blank lines into a single canonical separator."""
        normalized: t.MutableSequenceOf[str] = []
        previous_blank = False
        for line in lines:
            is_blank = not line.strip()
            if is_blank and previous_blank:
                continue
            normalized.append(line)
            previous_blank = is_blank
        return normalized

    def apply(self, rendered: str) -> t.Pair[str, t.StrSequence]:
        """Inject managed banner/markers and return updated TOML plus change messages."""
        changes: t.MutableSequenceOf[str] = []
        lines = rendered.splitlines()
        cleaned_lines, cleanup_changes = self._strip_managed_lines(lines)
        changes.extend(cleanup_changes)
        banner_lines = c.Infra.BANNER.splitlines()
        # NOTE (multi-agent, flext-wkii.17.9.2.1): banner injection owns the
        # leading separator so parse/render trivia cannot require a second pass.
        first_content = next(
            (index for index, line in enumerate(cleaned_lines) if line.strip()),
            len(cleaned_lines),
        )
        content_lines = cleaned_lines[first_content:]
        out: t.MutableSequenceOf[str] = [*banner_lines, ""]
        if lines[: len(banner_lines)] != banner_lines:
            changes.append("managed banner injected")
        emitted_markers: set[str] = set()
        for line in content_lines:
            stripped = line.strip()
            marker = self._marker_for_section(stripped)
            if marker and marker not in emitted_markers:
                out.append(marker)
                changes.append(f"marker injected for {stripped}")
                emitted_markers.add(marker)
            if stripped == "[project.optional-dependencies]" or stripped.startswith(
                "optional-dependencies.dev"
            ):
                self._inject_dev_markers(out, changes, emitted_markers)
            out.append(line)
            if stripped == "[tool.mypy]":
                out.extend(self._mypy_rationale_lines())
                changes.append("Mypy suppression rationales injected")
            elif stripped == "[tool.ruff.lint]":
                out.extend(self._ruff_rationale_lines())
                changes.append("Ruff suppression rationales injected")
            elif stripped == "[tool.pyright]":
                out.extend(self._pyright_rationale_lines())
                changes.append("Pyright suppression rationales injected")
        updated = "\n".join(self._collapse_blank_lines(out)).rstrip() + "\n"
        original = rendered.rstrip() + "\n"
        if updated == original:
            return (updated, [])
        return (updated, changes)


__all__: list[str] = ["FlextInfraInjectCommentsPhase"]
