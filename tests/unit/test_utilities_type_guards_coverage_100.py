"""Behavior contract for public type guards in metadata and dispatch flows.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from tests import m, p, tm, u

if TYPE_CHECKING:
    from tests import t


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
        tm.that(isinstance(payload, dict), truth=True)
        tm.that(isinstance(normalized_modes, list), truth=True)

        command_spec = m.GuardCheckSpec.model_validate({
            "starts": "sync",
            "contains": "users",
            "ends": "users",
            "none": False,
            "empty": False,
        })

        tm.that(u.dict_non_empty(metadata), truth=True)
        tm.that(u.matches_type(payload, "dict_non_empty"), truth=True)  # pyrefly: ignore
        tm.that(u.matches_type(envelope, "dict_non_empty"), truth=False)
        tm.that(u.matches_type(payload["tags"], "list_non_empty"), truth=True)  # pyrefly: ignore
        tm.that(u.matches_type(payload["attempt_count"], (int, float)), truth=True)  # pyrefly: ignore
        tm.that(u.matches_type(payload["command_name"], str), truth=True)  # pyrefly: ignore
        tm.that(u.string_non_empty(payload["correlation_id"]), truth=True)
        tm.that(u.chk(payload["command_name"], command_spec), truth=True)
        tm.that(u.chk(payload["attempt_count"], gte=1, lte=3, not_in=[0]), truth=True)
        tm.that(
            u.chk(payload["command_name"], gt="alpha", none=False, empty=False),
            truth=True,
        )
        tm.that(metadata["repository_root"], eq=str(repository_root))
        tm.ok(metadata["retry_window"] is None)
        tm.that(isinstance(payload["started_at"], str), truth=True)
        tm.that(
            datetime.fromisoformat(cast("str", payload["started_at"])),
            eq=envelope.started_at,
        )
        tm.that(set(cast("list[object]", normalized_modes)), eq={"delta", "full"})

    @staticmethod
    def test_public_guard_returns_normalized_values_defaults_and_failures() -> None:
        """Test public guard returns normalized values defaults and failures."""
        payload: t.JsonDict = {
            "command_name": "sync-users",
            "attempt_count": 2,
        }

        guarded_payload = u.guard(payload, dict, return_value=True)
        fallback_tags = u.guard(
            "invalid-tags",
            validator=u.list_like,
            default=["cli"],
            return_value=True,
        )
        guarded_attempt = u.guard(2, validator=(int, float), return_value=True)

        def raising_validator(value: t.JsonValue) -> bool:
            _ = value
            error_message = "invalid metadata"
            raise ValueError(error_message)

        failed_guard_result = u.guard(
            "sync-users",
            validator=raising_validator,
            return_value=True,
        )

        tm.that(guarded_payload, eq=payload)
        tm.that(fallback_tags, eq=["cli"])
        tm.that(guarded_attempt, eq=2)
        tm.that(isinstance(failed_guard_result, p.Result), truth=True)
        assert isinstance(failed_guard_result, p.Result)
        tm.fail(failed_guard_result)
        tm.that(
            cast("str", failed_guard_result.error),
            eq="Guard validation raised ValueError: invalid metadata",
        )
