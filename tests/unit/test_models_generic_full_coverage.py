"""Generic model tests with full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
"""

from __future__ import annotations

# pyright: reportMissingImports=false
from datetime import UTC, datetime, timedelta

import pytest
from flext_core import c, m, r, t, u


def test_operation_context_time_properties_and_flags() -> None:
    timestamp = datetime.now(UTC) - timedelta(minutes=6)
    context = m.OperationContext(
        correlation_id="corr-12345678",
        operation_id="op-12345678",
        timestamp=timestamp,
        source="api",
        user_id="u-1",
        tenant_id="tenant-a",
    )

    assert context.age_seconds > 0
    assert abs(context.age_minutes - (context.age_seconds / 60.0)) < 0.01
    assert context.is_recent is False
    assert context.formatted_timestamp == timestamp.isoformat()
    assert context.has_user_context is True
    assert context.has_tenant_context is True


def test_operation_context_context_summary_with_optional_parts() -> None:
    context = m.OperationContext(
        correlation_id="12345678-corr",
        operation_id="12345678-op",
        source="worker",
        user_id="alice",
        tenant_id="acme",
    )

    summary = context.context_summary
    assert "op:12345678" in summary
    assert "corr:12345678" in summary
    assert "src:worker" in summary
    assert "user:alice" in summary
    assert "tenant:acme" in summary


def test_operation_context_context_summary_without_optional_parts() -> None:
    context = m.OperationContext(
        correlation_id="abcdefgh-corr",
        operation_id="abcdefgh-op",
    )

    summary = context.context_summary
    assert "op:abcdefgh" in summary
    assert "corr:abcdefgh" in summary
    assert "src:" not in summary
    assert "user:" not in summary
    assert "tenant:" not in summary


def test_operation_context_with_metadata_merges_values() -> None:
    context = m.OperationContext(metadata=t.Dict(root={"a": 1, "b": "x"}))

    updated = context.with_metadata(b="y", c=True)

    assert updated is not context
    assert updated.metadata.root == {"a": 1, "b": "y", "c": True}
    assert context.metadata.root == {"a": 1, "b": "x"}
    assert updated.correlation_id == context.correlation_id
    assert updated.operation_id == context.operation_id


@pytest.mark.parametrize(
    ("child_operation_id", "expect_same_id"),
    [("child-1", True), (None, False)],
)
def test_operation_context_for_child_operation(
    child_operation_id: str | None,
    expect_same_id: bool,
) -> None:
    context = m.OperationContext(
        correlation_id="corr-fixed",
        operation_id="parent-op",
        source="svc",
        user_id="u1",
        tenant_id="t1",
    )

    child = context.for_child_operation(child_operation_id)

    assert child.correlation_id == context.correlation_id
    assert child.operation_id != context.operation_id
    if expect_same_id:
        assert child.operation_id == "child-1"
    else:
        assert isinstance(child.operation_id, str)
        assert len(child.operation_id) > 0
    assert child.timestamp >= context.timestamp
    assert child.metadata == context.metadata


def test_service_snapshot_active_and_healthy_flags() -> None:
    service = m.Service(status="active", health_status="healthy", name="api")

    assert service.is_active is True
    assert service.is_healthy is True


def test_service_snapshot_uptime_properties_and_formatting() -> None:
    service = m.Service(name="api", uptime_seconds=90061)

    assert service.uptime_hours is not None
    assert service.uptime_days is not None
    assert abs(service.uptime_hours - (90061 / 3600.0)) < 0.001
    assert abs(service.uptime_days - ((90061 / 3600.0) / 24.0)) < 0.001
    assert service.formatted_uptime == "1d 1h 1m"


def test_service_snapshot_unknown_uptime_and_endpoint_none() -> None:
    service = m.Service(name="api", host="localhost", port=None)

    assert service.formatted_uptime == "unknown"
    assert service.uptime_hours is None
    assert service.uptime_days is None
    assert service.endpoint_url is None
    assert service.health_check_age_minutes is None
    assert service.needs_health_check is True


def test_service_snapshot_endpoint_and_health_check_age() -> None:
    check_time = datetime.now(UTC) - timedelta(minutes=1)
    service = m.Service(
        name="api", host="127.0.0.1", port=8080, last_health_check=check_time
    )

    assert service.endpoint_url == "http://127.0.0.1:8080"
    assert service.health_check_age_minutes is not None
    assert service.health_check_age_minutes < c.Performance.HEALTH_CHECK_STALE_MINUTES
    assert service.needs_health_check is False


def test_service_snapshot_resource_summary_branches() -> None:
    with_metrics = m.Service(
        name="api", memory_usage_mb=128.25, cpu_usage_percent=43.75
    )
    no_metrics = m.Service(name="api")

    assert "RAM: 128.2MB" in with_metrics.resource_summary
    assert "CPU: 43.8%" in with_metrics.resource_summary
    assert no_metrics.resource_summary == "no metrics"


def test_service_snapshot_to_health_check_format() -> None:
    check_time = datetime.now(UTC)
    service = m.Service(
        name="api",
        version="2.0.0",
        status="active",
        health_status="healthy",
        uptime_seconds=61,
        last_health_check=check_time,
        metadata=t.Dict(root={"region": "us-east-1"}),
    )

    payload = service.to_health_check_format()

    assert payload["name"] == "api"
    assert payload["health"] == "healthy"
    assert payload["uptime"] == "1m"
    assert payload["timestamp"] == check_time.isoformat()
    assert payload["region"] == "us-east-1"


def test_configuration_properties_and_accessors() -> None:
    captured_at = datetime.now(UTC) - timedelta(hours=2)
    cfg = m.Configuration(
        config=t.Dict(root={"host": "localhost", "port": 8080}),
        captured_at=captured_at,
    )

    assert cfg.is_valid is True
    assert cfg.has_validation_errors is False
    assert cfg.validation_error_count == 0
    assert cfg.config_keys == ["host", "port"]
    assert cfg.config_size == 2
    assert cfg.age_minutes > c.Performance.RECENT_THRESHOLD_MINUTES
    assert cfg.is_recent is False
    assert cfg.formatted_captured_at == captured_at.isoformat()
    assert cfg.get("host") == "localhost"
    assert cfg.get("missing", "default") == "default"
    assert cfg.has_key("port") is True
    assert cfg.has_key("missing") is False


def test_configuration_environment_conversion_and_required_keys() -> None:
    cfg = m.Configuration(config=t.Dict(root={"host": "localhost", "port": 8080}))

    assert cfg.to_environment_variables() == {"HOST": "localhost", "PORT": "8080"}
    assert cfg.to_environment_variables("APP_") == {
        "APP_HOST": "localhost",
        "APP_PORT": "8080",
    }
    assert cfg.validate_required_keys(["host", "port", "mode"]) == ["mode"]


def test_configuration_with_validation_errors_returns_new_instance() -> None:
    cfg = m.Configuration(config=t.Dict(root={"key": "value"}))

    failed = cfg.with_validation_errors(["missing key", "bad format"])

    assert failed is not cfg
    assert failed.validation_errors == ["missing key", "bad format"]
    assert failed.is_valid is False
    assert failed.has_validation_errors is True
    assert failed.validation_error_count == 2


def test_health_counts_and_percentage_with_empty_checks() -> None:
    health = m.Health()

    assert health.total_checks == 0
    assert health.healthy_checks_count == 0
    assert health.unhealthy_checks_count == 0
    assert health.health_percentage == pytest.approx(100.0)
    assert health.severity_level == "unknown"


def test_health_counts_lists_and_status_summary() -> None:
    health = m.Health(
        checks=t.Dict(root={"db": True, "cache": False, "queue": True}),
        details=t.Dict(root={"db": "ok", "cache": "timeout"}),
    )

    assert health.total_checks == 3
    assert health.healthy_checks_count == 2
    assert health.unhealthy_checks_count == 1
    assert abs(health.health_percentage - ((2 / 3) * 100.0)) < 0.001
    assert health.unhealthy_checks == ["cache"]
    assert health.healthy_checks == ["db", "queue"]
    assert health.status_summary == "2/3 checks passed"
    assert health.get_check_detail("db") == "ok"
    assert health.get_check_detail("missing") is None


@pytest.mark.parametrize(
    ("checks", "expected"),
    [
        ({"db": True, "cache": True}, "healthy"),
        ({"db": False, "cache": False, "queue": True}, "critical"),
    ],
)
def test_health_severity_level_branches(
    checks: dict[str, bool],
    expected: str,
) -> None:
    typed_checks: dict[str, t.GeneralValueType] = dict(checks.items())
    health = m.Health(checks=t.Dict(root=typed_checks))

    assert health.severity_level == expected


def test_health_severity_level_warning_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(c.Performance, "FAILURE_RATE_WARNING_THRESHOLD", 0.5)
    health = m.Health(checks=t.Dict(root={"db": True, "cache": False, "queue": True}))

    assert health.severity_level == "warning"


def test_health_age_recent_and_monitoring_export() -> None:
    checked_at = datetime.now(UTC) - timedelta(seconds=10)
    health = m.Health(
        healthy=False,
        checks=t.Dict(root={"db": False}),
        checked_at=checked_at,
        service_name="svc",
        service_version="1.2.3",
        duration_ms=17.5,
        metadata=t.Dict(root={"cluster": "a"}),
    )

    assert health.age_seconds >= 0
    assert health.is_recent is True
    assert health.formatted_checked_at == checked_at.isoformat()

    monitoring = health.to_monitoring_format()
    assert monitoring.root["status"] == "down"
    assert monitoring.root["severity"] == "critical"
    assert monitoring.root["cluster"] == "a"
    assert monitoring.root["duration_ms"] == pytest.approx(17.5)


def test_health_with_additional_check_with_and_without_detail() -> None:
    base = m.Health(
        healthy=True, checks=t.Dict(root={"db": True}), details=t.Dict(root={})
    )

    with_detail = base.with_additional_check("cache", False, detail="timeout")
    without_detail = with_detail.with_additional_check("queue", True)

    assert with_detail.checks.root == {"db": True, "cache": False}
    assert with_detail.details.root == {"cache": "timeout"}
    assert with_detail.healthy is False
    assert without_detail.details.root == {"cache": "timeout"}


def test_operation_progress_core_rates_and_remaining() -> None:
    op = m.Operation(
        success_count=8,
        failure_count=1,
        skipped_count=1,
        warning_count=0,
        estimated_total=20,
    )

    assert op.total_count == 10
    assert op.success_rate == pytest.approx(0.8)
    assert op.failure_rate == pytest.approx(0.1)
    assert op.completion_percentage == pytest.approx(50.0)
    assert op.remaining_count == 10
    assert op.is_complete is False


def test_operation_progress_remaining_and_complete_without_estimate() -> None:
    op = m.Operation(success_count=1)

    assert op.remaining_count is None
    assert op.is_complete is False
    assert op.completion_percentage == pytest.approx(0.0)


def test_operation_progress_completion_percentage_with_zero_estimate() -> None:
    op = m.Operation(success_count=5, estimated_total=0)

    assert op.completion_percentage == pytest.approx(0.0)


def test_operation_progress_duration_items_and_estimated_remaining() -> None:
    start = datetime.now(UTC) - timedelta(seconds=20)
    end = datetime.now(UTC) - timedelta(seconds=5)
    op = m.Operation(
        success_count=9,
        failure_count=1,
        estimated_total=20,
        start_time=start,
        last_update=end,
    )

    assert op.duration_seconds is not None
    assert abs(op.duration_seconds - 15.0) <= 3.0
    assert op.items_per_second is not None
    assert op.items_per_second > 0
    assert op.estimated_time_remaining_seconds is not None
    assert op.estimated_time_remaining_seconds > 0


def test_operation_progress_duration_and_throughput_unavailable() -> None:
    op = m.Operation(success_count=2, estimated_total=10)

    assert op.duration_seconds is None
    assert op.items_per_second is None
    assert op.estimated_time_remaining_seconds is None


def test_operation_progress_error_warning_and_status_summary() -> None:
    op = m.Operation(
        success_count=3,
        failure_count=2,
        skipped_count=1,
        warning_count=1,
        estimated_total=10,
    )

    assert op.has_errors is True
    assert op.has_warnings is True
    summary = op.status_summary
    assert "3 success" in summary
    assert "2 failed" in summary
    assert "1 skipped" in summary
    assert "1 warnings" in summary
    assert ".1f" in summary


def test_operation_progress_mutation_methods_update_counts_and_timestamps() -> None:
    op = m.Operation()

    op.record_success()
    op.record_failure()
    op.record_skip()
    op.record_warning()
    op.record_retry()
    op.set_current_item("item-42")

    assert op.success_count == 1
    assert op.failure_count == 1
    assert op.skipped_count == 1
    assert op.warning_count == 1
    assert op.retry_count == 1
    assert op.current_item == "item-42"
    assert op.last_update is not None


def test_operation_progress_start_operation_sets_runtime_fields() -> None:
    op = m.Operation()

    op.start_operation(name="sync", estimated_total=5)

    assert op.operation_name == "sync"
    assert op.estimated_total == 5
    assert op.start_time is not None
    assert op.last_update == op.start_time


def test_operation_progress_report_export() -> None:
    start = datetime.now(UTC) - timedelta(seconds=10)
    end = datetime.now(UTC) - timedelta(seconds=5)
    op = m.Operation(
        success_count=3,
        failure_count=1,
        skipped_count=0,
        warning_count=0,
        estimated_total=8,
        current_item="x",
        operation_name="ingest",
        start_time=start,
        last_update=end,
    )

    report = op.to_progress_report().root

    assert report["operation"] == "ingest"
    assert report["total_processed"] == 4
    assert report["success_rate"] == "0.750"
    assert report["completion_percentage"] == "50.0"
    assert report["items_per_second"] is not None
    assert report["estimated_time_remaining"] is not None
    assert report["is_complete"] is False


def test_conversion_properties_basic_counts_and_rates() -> None:
    conv = m.Conversion(
        converted=[1, 2],
        errors=["e1"],
        warnings=["w1"],
        skipped=[3],
        total_input_count=10,
    )

    assert conv.has_errors is True
    assert conv.has_warnings is True
    assert conv.converted_count == 2
    assert conv.skipped_count == 1
    assert conv.error_count == 1
    assert conv.warning_count == 1
    assert conv.total_processed_count == 3
    assert abs(conv.success_rate - (2 / 3)) < 0.001
    assert conv.completion_percentage == pytest.approx(30.0)


def test_conversion_duration_items_per_second_and_completion_flags() -> None:
    now = datetime.now(UTC)
    incomplete = m.Conversion(start_time=now - timedelta(seconds=5), end_time=None)
    complete = m.Conversion(
        converted=[1, 2, 3],
        skipped=[4],
        start_time=now - timedelta(seconds=8),
        end_time=now - timedelta(seconds=2),
    )

    assert incomplete.duration_seconds is None
    assert incomplete.items_per_second is None
    assert incomplete.is_complete is False
    assert complete.duration_seconds is not None
    assert abs(complete.duration_seconds - 6.0) <= 1.2
    assert complete.items_per_second is not None
    assert complete.items_per_second > 0
    assert complete.is_complete is True


def test_conversion_status_summary_and_detailed_summary_optional_lines() -> None:
    now = datetime.now(UTC)
    conv = m.Conversion(
        converted=[1, 2],
        errors=["e1"],
        warnings=["w1"],
        skipped=[3],
        source_format="csv",
        target_format="json",
        start_time=now - timedelta(seconds=10),
        end_time=now - timedelta(seconds=5),
        total_input_count=10,
    )

    status = conv.status_summary
    summary = conv.conversion_summary
    assert "2 converted" in status
    assert "1 errors" in status
    assert ".1f" in status
    assert "Conversion: csv" in summary
    assert "Status: Complete" in summary
    assert ".1%" in summary
    assert ".2f" in summary
    assert ".1f" in summary


def test_conversion_summary_in_progress_and_unknown_formats() -> None:
    conv = m.Conversion()

    summary = conv.conversion_summary
    assert "Conversion: unknown" in summary
    assert "Status: In Progress" in summary


def test_conversion_add_converted_and_error_metadata_append_paths() -> None:
    conv = m.Conversion()

    conv.add_converted("ok-1")
    conv.add_error("bad-1", item="item-a")
    conv.add_error("bad-2", item="item-b")
    conv.add_error("bad-3")

    assert conv.converted == ["ok-1"]
    assert conv.errors == ["bad-1", "bad-2", "bad-3"]
    assert conv.metadata.root["failed_items"] == ["item-a", "item-b"]


def test_conversion_add_warning_metadata_append_paths() -> None:
    conv = m.Conversion()

    conv.add_warning("warn-1", item="item-a")
    conv.add_warning("warn-2", item="item-b")
    conv.add_warning("warn-3")

    assert conv.warnings == ["warn-1", "warn-2", "warn-3"]
    assert conv.metadata.root["warning_items"] == ["item-a", "item-b"]


def test_conversion_add_skipped_skip_reason_upsert_paths() -> None:
    conv = m.Conversion()

    conv.add_skipped("item-a", reason="empty")
    conv.add_skipped("item-b", reason="invalid")
    conv.add_skipped("item-c")

    assert conv.skipped == ["item-a", "item-b", "item-c"]
    assert conv.metadata.root["skip_reasons"] == {
        "item-a": "empty",
        "item-b": "invalid",
    }


def test_conversion_start_and_complete_methods() -> None:
    conv = m.Conversion()

    conv.start_conversion(
        source_format="xml", target_format="json", total_input_count=99
    )
    conv.complete_conversion()

    assert conv.source_format == "xml"
    assert conv.target_format == "json"
    assert conv.total_input_count == 99
    assert conv.start_time is not None
    assert conv.end_time is not None
    assert conv.end_time >= conv.start_time


def test_conversion_report_export() -> None:
    now = datetime.now(UTC)
    conv = m.Conversion(
        converted=[1, 2],
        errors=["e1"],
        warnings=["w1"],
        skipped=[3],
        source_format="csv",
        target_format="json",
        total_input_count=10,
        start_time=now - timedelta(seconds=10),
        end_time=now - timedelta(seconds=5),
    )

    report = conv.to_conversion_report().root

    assert report["source_format"] == "csv"
    assert report["target_format"] == "json"
    assert report["converted_count"] == 2
    assert report["error_count"] == 1
    assert report["warning_count"] == 1
    assert report["skipped_count"] == 1
    assert report["success_rate"] == "0.667"
    assert report["completion_percentage"] == "30.0"
    assert report["is_complete"] is True
    assert report["has_errors"] is True
    assert report["has_warnings"] is True


def test_canonical_aliases_are_available() -> None:
    result = r[str].ok("ok")
    value = t.Dict(root={"k": "v"})
    assert result.value == "ok"
    assert value.root == {"k": "v"}
    assert c.Performance.RECENT_THRESHOLD_SECONDS > 0
    assert hasattr(u, "validated")
