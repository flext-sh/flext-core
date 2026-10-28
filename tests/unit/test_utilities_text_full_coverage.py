"""Behavior contract for public text helpers in real bootstrap workflows.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

from pathlib import Path

import pytest

from tests import c, m, tm, u, utilities


class TestsFlextUtilitiesText(utilities.u.Tests.Contract):
    """Tests for ``FlextUtilitiesText``."""

    @staticmethod
    @pytest.mark.parametrize(
        ("value", "message"),
        [
            pytest.param(None, c.ERR_TEXT_NONE_NOT_ALLOWED, id="none"),
            pytest.param("", c.ERR_TEXT_EMPTY_NOT_ALLOWED, id="empty"),
            pytest.param("   ", c.ERR_TEXT_EMPTY_NOT_ALLOWED, id="blank"),
        ],
    )
    def test_public_text_helpers_reject_blank_bootstrap_inputs(
        value: str | None,
        message: str,
    ) -> None:
        """Bootstrap inputs must fail fast when text is absent or blank."""
        with pytest.raises(ValueError, match=message):
            u.safe_string(value)

    @staticmethod
    @pytest.mark.parametrize(
        ("raw", "expected_id", "expected_key"),
        [
            pytest.param(
                "  Fleet Sync_App v2  ",
                "fleet-sync-app-v2",
                "fleetsyncappv2",
                id="mixed-space-underscore",
            ),
            pytest.param("A B", "a-b", "ab", id="uppercase-space"),
            pytest.param("__x__", "--x--", "x", id="underscore-padding"),
        ],
    )
    def test_public_text_helpers_derive_stable_identifiers(
        raw: str,
        expected_id: str,
        expected_key: str,
    ) -> None:
        """format_app_id and normalize_alnum expose a deterministic contract."""
        cleaned = u.safe_string(raw)

        app_id = u.format_app_id(cleaned)
        normalized_key = u.normalize_alnum(cleaned)

        tm.that(app_id, eq=expected_id)
        tm.that(app_id, eq=app_id.lower())
        tm.that(" " not in app_id, truth=True)
        tm.that("_" not in app_id, truth=True)
        tm.that(normalized_key, eq=expected_key)
        tm.that(normalized_key.isalnum() or not normalized_key, truth=True)

    @staticmethod
    @pytest.mark.parametrize(
        "value",
        [
            pytest.param("  spaced  ", id="surrounding-space"),
            pytest.param("a b", id="inner-space"),
            pytest.param("Fleet Sync_App v2", id="app-name"),
        ],
    )
    def test_public_text_helpers_are_idempotent(value: str) -> None:
        """Re-applying the helpers to their own output is a fixed point."""
        cleaned = u.safe_string(value)
        tm.that(u.safe_string(cleaned), eq=cleaned)

        app_id = u.format_app_id(cleaned)
        tm.that(u.format_app_id(app_id), eq=app_id)

        normalized_key = u.normalize_alnum(cleaned)
        tm.that(u.normalize_alnum(normalized_key), eq=normalized_key)

    @staticmethod
    def test_public_text_helpers_prepare_and_persist_app_manifest(
        tmp_path: Path,
    ) -> None:
        """App bootstrap uses the public helpers to normalize and persist text."""
        raw_name = "  Fleet Sync_App v2  "
        cleaned_name = u.safe_string(raw_name)
        app_id = u.format_app_id(cleaned_name)
        normalized_key = u.normalize_alnum(cleaned_name)
        manifest_path = tmp_path / "app.env"
        manifest_content = "\n".join([
            f"APP_NAME={cleaned_name}",
            f"APP_ID={app_id}",
            f"APP_KEY={normalized_key}",
        ])

        u.write_file(manifest_path, manifest_content)

        snapshot = m.Tests.ManifestSnapshot(
            app_id=app_id,
            normalized_key=normalized_key,
            manifest_path=str(manifest_path),
            manifest_content=manifest_path.read_text(encoding=c.DEFAULT_ENCODING),
        )

        tm.that(snapshot.app_id, eq="fleet-sync-app-v2")
        tm.that(snapshot.normalized_key, eq="fleetsyncappv2")
        tm.that(Path(snapshot.manifest_path), eq=manifest_path)
        tm.that(snapshot.manifest_content, eq=manifest_content)
        tm.that(" " not in snapshot.app_id, truth=True)
        tm.that("_" not in snapshot.app_id, truth=True)
        tm.that(snapshot.normalized_key.isalnum(), truth=True)
