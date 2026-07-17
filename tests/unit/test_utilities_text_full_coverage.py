"""Behavior contract for public text helpers in real bootstrap workflows."""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_core import u as core_u
from tests import c
from tests import m
from tests import u as test_u


class TestsFlextUtilitiesText(test_u.Tests.Contract):
    @pytest.mark.parametrize(
        ("value", "message"),
        [
            pytest.param(None, c.ERR_TEXT_NONE_NOT_ALLOWED, id="none"),
            pytest.param("", c.ERR_TEXT_EMPTY_NOT_ALLOWED, id="empty"),
            pytest.param("   ", c.ERR_TEXT_EMPTY_NOT_ALLOWED, id="blank"),
        ],
    )
    def test_public_text_helpers_reject_blank_bootstrap_inputs(
        self,
        value: str | None,
        message: str,
    ) -> None:
        """Bootstrap inputs must fail fast when text is absent or blank."""
        with pytest.raises(ValueError, match=message):
            core_u.safe_string(value)

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
        self,
        raw: str,
        expected_id: str,
        expected_key: str,
    ) -> None:
        """format_app_id and normalize_alnum expose a deterministic contract."""
        cleaned = core_u.safe_string(raw)

        app_id = core_u.format_app_id(cleaned)
        normalized_key = core_u.normalize_alnum(cleaned)

        assert app_id == expected_id
        assert app_id == app_id.lower()
        assert " " not in app_id
        assert "_" not in app_id
        assert normalized_key == expected_key
        assert normalized_key.isalnum() or normalized_key == ""

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param("  spaced  ", id="surrounding-space"),
            pytest.param("a b", id="inner-space"),
            pytest.param("Fleet Sync_App v2", id="app-name"),
        ],
    )
    def test_public_text_helpers_are_idempotent(self, value: str) -> None:
        """Re-applying the helpers to their own output is a fixed point."""
        cleaned = core_u.safe_string(value)
        assert core_u.safe_string(cleaned) == cleaned

        app_id = core_u.format_app_id(cleaned)
        assert core_u.format_app_id(app_id) == app_id

        normalized_key = core_u.normalize_alnum(cleaned)
        assert core_u.normalize_alnum(normalized_key) == normalized_key

    def test_public_text_helpers_prepare_and_persist_app_manifest(
        self,
        tmp_path: Path,
    ) -> None:
        """App bootstrap uses the public helpers to normalize and persist text."""
        raw_name = "  Fleet Sync_App v2  "
        cleaned_name = core_u.safe_string(raw_name)
        app_id = core_u.format_app_id(cleaned_name)
        normalized_key = core_u.normalize_alnum(cleaned_name)
        manifest_path = tmp_path / "app.env"
        manifest_content = "\n".join([
            f"APP_NAME={cleaned_name}",
            f"APP_ID={app_id}",
            f"APP_KEY={normalized_key}",
        ])

        core_u.write_file(manifest_path, manifest_content)

        snapshot = m.Tests.ManifestSnapshot(
            app_id=app_id,
            normalized_key=normalized_key,
            manifest_path=str(manifest_path),
            manifest_content=manifest_path.read_text(encoding=c.DEFAULT_ENCODING),
        )

        assert snapshot.app_id == "fleet-sync-app-v2"
        assert snapshot.normalized_key == "fleetsyncappv2"
        assert Path(snapshot.manifest_path) == manifest_path
        assert snapshot.manifest_content == manifest_content
        assert " " not in snapshot.app_id
        assert "_" not in snapshot.app_id
        assert snapshot.normalized_key.isalnum()
