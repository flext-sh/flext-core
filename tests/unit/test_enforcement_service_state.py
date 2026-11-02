"""Exercise native service-state roles in a real, statically imported consumer.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from flext_tests import m, tm, u


class TestsFlextCoreEnforcementServiceState:
    """Validate the actual consumer process without changing Python import paths."""

    @staticmethod
    @pytest.mark.parametrize(
        ("case", "value"),
        [
            ("state", "state"),
            ("validated", "STATE"),
            ("helper", None),
            ("foreign", None),
            ("behavior", None),
            ("inherited_behavior", None),
            ("property", None),
            ("inherited_foreign", None),
            ("empty", None),
            ("lookalike", None),
            ("shared_foreign", None),
            ("service_as_state", None),
        ],
    )
    def test_native_service_state(
        tmp_path: Path,
        case: str,
        value: str | None,
    ) -> None:
        """Run authored static imports in an owned CWD through the native runner."""
        checkout = Path(__file__).resolve().parents[2]
        source = checkout / "tests" / "fixtures" / "native_mro_consumer.json"
        catalog = m.Tests.Payload.model_validate_json(
            source.read_text(encoding="utf-8"),
        )
        tm.that(catalog.kind, eq="mapping")
        tm.that(catalog.entries["schema"].atom, eq="flext_tests.m.Tests.Payload")
        tm.that(catalog.entries["package"].atom, eq="native_mro_consumer")
        tm.that(catalog.entries["source_count"].atom, eq=9)
        sources = catalog.entries["sources"]
        tm.that(sources.kind, eq="mapping")
        content = u.type_adapter(dict[str, str]).validate_python(
            u.Tests.FlextTestsPayloadUtilities.to_normalized_value(sources),
        )
        tm.that(len(content), eq=9)
        with u.Cli.files_context(content, directory=tmp_path) as created:
            tm.that(frozenset(created), eq=frozenset(sources.entries))
            for name, path in created.items():
                tm.that(path.read_bytes(), eq=content[name].encode("utf-8"))
            output = tm.ok(
                u.Cli.run_raw(
                    (sys.executable, "-m", "native_mro_consumer.cli"),
                    cwd=tmp_path,
                )
            )
            tm.that(u.Cli.process_succeeded(output.outcome), eq=True, msg=output.stderr)
            tm.that(output.outcome.timed_out, eq=False)
            tm.that(output.outcome.forwarded_signal, eq=None)
            for name, path in created.items():
                tm.that(path.read_bytes(), eq=content[name].encode("utf-8"))
            report = u.Tests.FlextTestsPayloadUtilities.to_payload(
                m.ConfigMap.model_validate_json(output.stdout),
            )
            tm.that(report.entries["python_prefix"].atom, eq=sys.prefix)
            tm.that(
                report.entries["core_service_source"].atom,
                eq=str(checkout / "src/flext_core/service.py"),
            )
            tm.that(report.entries["cwd"].atom, eq=str(tmp_path))
            outcomes = report.entries["cases"].items
            tm.that(len(outcomes), eq=12)
            outcome = next(
                item for item in outcomes if item.entries["name"].atom == case
            )
            tm.that(
                outcome.entries["target_module"].atom,
                eq="native_mro_consumer.base",
            )
            qualname = u.type_adapter(str).validate_python(
                outcome.entries["target_qualname"].atom,
            )
            tm.that("<locals>" in qualname, eq=False)
            rules = frozenset({"ENFORCE-047", "ENFORCE-049"})
            expected = rules if value is None else frozenset[str]()
            actual = u.type_adapter(frozenset[str]).validate_python(
                u.Tests.FlextTestsPayloadUtilities.to_normalized_value(
                    outcome.entries["rule_ids"],
                ),
            )
            tm.that(actual & rules, eq=expected)
            tm.that(outcome.entries["value"].atom, eq=value)
