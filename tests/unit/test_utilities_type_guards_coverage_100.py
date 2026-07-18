"""Behavior contract for public type guards in metadata and dispatch flows."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from flext_tests import tm

from flext_core import u
from tests import m

from tests import t


class TestsFlextCoreUtilitiesTypeGuards:
    """Behavior contract for public guard helpers on runtime metadata."""

    def test_public_type_guards_validate_normalized_dispatch_metadata(self) -> None:
        envelope = m.Tests.DispatchEnvelope(
            command_name="sync-users",
            correlation_id="corr_12345678",
            attempt_count=2,
            tags=("ldap", "prod"),
            started_at=datetime(2026, 5, 5, 12, 0, tzinfo=UTC),
        )
        normalized_payload = u.normalize_to_metadata(envelope)
        normalized_workspace_root = u.normalize_to_metadata(Path("/tmp/flext"))
        normalized_retry_window = u.normalize_to_metadata(None)
        normalized_modes = u.normalize_to_metadata({"delta", "full"})

        metadata: dict[str, t.JsonValue] = {
            "payload": normalized_payload,
            "workspace_root": normalized_workspace_root,
            "retry_window": normalized_retry_window,
            "modes": normalized_modes,
        }

        payload = metadata["payload"]
        tm.that(payload, is_=dict)
        tm.that(normalized_modes, is_=list)

        command_spec = m.GuardCheckSpec.model_validate({
            "starts": "sync",
            "contains": "users",
            "ends": "users",
            "none": False,
            "empty": False,
        })

        tm.that(u.dict_non_empty(metadata), eq=True)
        tm.that(u.matches_type(payload, "dict_non_empty"), eq=True)
        tm.that(u.matches_type(envelope, "dict_non_empty"), eq=False)
        tm.that(u.matches_type(payload["tags"], "list_non_empty"), eq=True)
        tm.that(u.matches_type(payload["attempt_count"], (int, float)), eq=True)
        tm.that(u.matches_type(payload["command_name"], str), eq=True)
        tm.that(u.string_non_empty(payload["correlation_id"]), eq=True)
        tm.that(u.chk(payload["command_name"], command_spec), eq=True)
        tm.that(u.chk(payload["attempt_count"], gte=1, lte=3, not_in=[0]), eq=True)
        tm.that(u.chk(payload["command_name"], gt="alpha", none=False, empty=False), eq=True)
        tm.that(metadata["workspace_root"], eq="/tmp/flext")
        tm.that(metadata["retry_window"], eq=None)
        tm.that(payload["started_at"], is_=str)
        tm.that(datetime.fromisoformat(payload["started_at"]), eq=envelope.started_at)
        tm.that(set(normalized_modes), eq={"delta", "full"})

    def test_public_guard_returns_normalized_values_defaults_and_failures(self) -> None:
        payload: dict[str, t.JsonValue] = {
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

        tm.that(isinstance(
            failed_guard_result,
            (bool, dict, list, str, int, float, type(None)),
        ), eq=False)
        failed_guard = failed_guard_result

        tm.that(cast("dict[str, t.JsonValue]", guarded_payload), eq=payload)
        tm.that(cast("list[str]", fallback_tags), eq=["cli"])
        tm.that(cast("int", guarded_attempt), eq=2)
        tm.fail(failed_guard)
        tm.that(failed_guard.error, eq="Guard validation raised an exception")
