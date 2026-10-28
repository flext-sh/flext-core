"""Behavior contract for public type guards in metadata and dispatch flows.

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

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from tests import c, m, t, tm, u


class TestsFlextCoreUtilitiesTypeGuards:
    """Behavior contract for public guard helpers on runtime metadata."""

    @staticmethod
    def test_public_type_guards_validate_normalized_dispatch_metadata(
        tmp_path: Path,
    ) -> None:
        """Test public type guards validate normalized dispatch metadata."""
        repository_root = tmp_path / "flext"
        repository_root.mkdir()
        envelope = m.Tests.DispatchEnvelope(
            command_name="sync-users",
            correlation_id="corr_12345678",
            attempt_count=2,
            tags=("ldap", "prod"),
            started_at=datetime(2026, 5, 5, 12, 0, tzinfo=UTC),
        )
        normalized_payload = u.normalize_to_json_value(envelope)
        normalized_repository_root = u.normalize_to_json_value(repository_root)
        normalized_retry_window = u.normalize_to_json_value(None)
        normalized_modes = u.normalize_to_json_value({"delta", "full"})

        metadata: t.JsonDict = {
            "payload": normalized_payload,
            "repository_root": normalized_repository_root,
            "retry_window": normalized_retry_window,
            "modes": normalized_modes,
        }

        payload = cast("t.JsonDict", metadata["payload"])
        tm.that(payload, is_=dict)
        tm.that(isinstance(normalized_modes, list), truth=True)

        tm.that(u.dict_non_empty(metadata), truth=True)
        tm.that(u.matches_type(payload, "dict_non_empty"), truth=True)
        tm.that(u.matches_type(envelope, "dict_non_empty"), truth=False)
        tm.that(u.matches_type(payload["tags"], "list_non_empty"), truth=True)
        tm.that(u.matches_type(payload["attempt_count"], (int, float)), truth=True)
        tm.that(u.matches_type(payload["command_name"], str), truth=True)
        tm.that(u.string_non_empty(payload[c.ContextKey.CORRELATION_ID]), truth=True)
        tm.that(metadata["repository_root"], eq=str(repository_root))
        tm.ok(metadata["retry_window"] is None)
        tm.that(isinstance(payload["started_at"], str), truth=True)
        tm.that(
            datetime.fromisoformat(cast("str", payload["started_at"])),
            eq=envelope.started_at,
        )
        tm.that(set(cast("list[object]", normalized_modes)), eq={"delta", "full"})
