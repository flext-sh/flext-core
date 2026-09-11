"""Centralized constants for the release subpackage.

The release protocol derives every version bump from the Conventional Commits
subject that a merged pull request leaves on its merge commit, writes the
version only through the protocol, and identifies a release by one tag shape.
These are external contracts (Conventional Commits, PEP 440, Git), so they are
constants rather than configuration.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraConstantsRelease:
    """Release infrastructure constants."""

    VERSION_RELEASE_SEGMENTS: Final[int] = 3

    class ReleasePhase(StrEnum):
        """Canonical release phases; each one is one Make ``WHAT`` selector."""

        PLAN = "plan"
        VERSION = "version"
        TAG = "tag"
        BUILD = "build"
        PUBLISH = "publish"

    class VersionBump(StrEnum):
        """Canonical semantic-version bump kinds, ordered by significance."""

        NONE = "none"
        PATCH = "patch"
        MINOR = "minor"
        MAJOR = "major"

    VERSION_RE: Final[t.RegexPattern] = re.compile(
        r"^version\s*=\s*['\"](.+?)['\"]", re.MULTILINE
    )
    CONVENTIONAL_SUBJECT_RE: Final[t.RegexPattern] = re.compile(
        r"^(?P<type>[a-z]+)(?:\([^)]+\))?(?P<breaking>!)?: \S"
    )
    "Conventional Commits subject: ``type(scope)!: description``."
    PULL_REQUEST_MERGE_SUBJECT_RE: Final[t.RegexPattern] = re.compile(
        r"^Merge pull request #\d+\b"
    )
    "GitHub's default merge subject, which carries no release information."
    TAG_FORMAT: Final[str] = "v{version}"
    RELEASE_BRANCH: Final[str] = "release/next"
    "One bot-owned lane per repository; the open release pull request lives here."
    RELEASE_COMMIT_SUBJECT: Final[str] = "chore(release): v{version}"
    RELEASE_COMMIT_SUBJECT_RE: Final[t.RegexPattern] = re.compile(
        r"^chore\(release\): v(?P<version>\S+?)(?: \(#\d+\))?$"
    )
    "The release commit as Git carries it: GitHub appends ` (#N)` when merging."
    RELEASE_PLAN_FILENAME: Final[str] = "plan.json"
    RELEASE_NOTES_FILENAME: Final[str] = "RELEASE_NOTES.md"
    RELEASE_REPORT_FILENAME: Final[str] = "build-report.json"
    GH: Final[str] = "gh"


__all__: list[str] = ["FlextInfraConstantsRelease"]
