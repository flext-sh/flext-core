"""Monitoring configuration patterns - consolidated from multiple projects.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module consolidates monitoring configuration patterns found across:
- flext-observability, flext-api, flext-grpc, flext-web, flext-auth, flext-quality
"""

from __future__ import annotations

from pydantic import Field
from pydantic import field_validator

from flext_core.config.base import BaseConfig


class MonitoringConfig(BaseConfig):
    """Monitoring configuration - consolidated pattern from 6+ projects."""

    # Health check settings
    health_enabled: bool = Field(
        default=True,
        description="Enable health check endpoint",
    )
    health_endpoint: str = Field(
        default="/health",
        description="Health check endpoint path",
    )
    health_interval: int = Field(
        default=30,
        ge=1,
        le=3600,
        description="Health check interval in seconds",
    )
    health_timeout: int = Field(
        default=10,
        ge=1,
        le=300,
        description="Health check timeout in seconds",
    )

    # Metrics settings
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    metrics_endpoint: str = Field(
        default="/metrics",
        description="Metrics endpoint path",
    )
    metrics_interval: int = Field(
        default=30,
        ge=1,
        le=3600,
        description="Metrics collection interval in seconds",
    )
    metrics_namespace: str = Field(
        default="flext",
        description="Metrics namespace/prefix",
    )

    # Logging settings
    log_level: str = Field(default="INFO", description="Logging level")
    access_log_enabled: bool = Field(default=True, description="Enable access logging")
    structured_logging: bool = Field(
        default=True,
        description="Enable structured logging",
    )
    log_format: str = Field(default="json", description="Log format")

    # Tracing settings
    tracing_enabled: bool = Field(
        default=False,
        description="Enable distributed tracing",
    )
    trace_sample_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Trace sampling rate",
    )
    jaeger_endpoint: str | None = Field(default=None, description="Jaeger endpoint URL")

    # Alerting settings
    alerting_enabled: bool = Field(default=False, description="Enable alerting")
    alert_manager_url: str | None = Field(default=None, description="Alert Manager URL")
    webhook_url: str | None = Field(default=None, description="Webhook URL for alerts")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed_levels:
            msg = f"Log level must be one of: {allowed_levels}"
            raise ValueError(msg)
        return v.upper()

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Validate log format."""
        allowed_formats = {"json", "text", "logfmt"}
        if v.lower() not in allowed_formats:
            msg = f"Log format must be one of: {allowed_formats}"
            raise ValueError(msg)
        return v.lower()

    @field_validator("health_endpoint", "metrics_endpoint")
    @classmethod
    def validate_endpoint_path(cls, v: str) -> str:
        """Validate endpoint path format."""
        if not v.startswith("/"):
            v = f"/{v}"
        return v

    @field_validator("jaeger_endpoint", "alert_manager_url", "webhook_url")
    @classmethod
    def validate_url(cls, v: str | None) -> str | None:
        """Validate URL format."""
        if v is None:
            return v
        if not v.startswith(("http://", "https://")):
            msg = "URL must start with http:// or https://"
            raise ValueError(msg)
        return v.rstrip("/")


class PrometheusConfig(BaseConfig):
    """Prometheus-specific monitoring configuration."""

    # Prometheus settings
    prometheus_enabled: bool = Field(
        default=True,
        description="Enable Prometheus metrics",
    )
    prometheus_port: int = Field(
        default=9090,
        ge=1,
        le=65535,
        description="Prometheus port",
    )
    prometheus_path: str = Field(
        default="/metrics",
        description="Prometheus metrics path",
    )

    # Scrape settings
    scrape_interval: int = Field(
        default=15,
        ge=5,
        le=300,
        description="Scrape interval in seconds",
    )
    scrape_timeout: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Scrape timeout in seconds",
    )

    # Labels and metadata
    instance_name: str = Field(default="flext", description="Instance name for metrics")
    job_name: str = Field(default="flext", description="Job name for metrics")
    additional_labels: dict[str, str] = Field(
        default_factory=dict,
        description="Additional metric labels",
    )

    @field_validator("prometheus_path")
    @classmethod
    def validate_prometheus_path(cls, v: str) -> str:
        """Validate Prometheus path format."""
        if not v.startswith("/"):
            v = f"/{v}"
        return v


class GrafanaConfig(BaseConfig):
    """Grafana-specific monitoring configuration."""

    # Grafana settings
    grafana_enabled: bool = Field(
        default=False,
        description="Enable Grafana integration",
    )
    grafana_url: str | None = Field(default=None, description="Grafana URL")
    grafana_api_key: str | None = Field(
        default=None,
        description="Grafana API key",
        json_schema_extra={"secret": True},
    )

    # Dashboard settings
    dashboard_uid: str | None = Field(default=None, description="Default dashboard UID")
    organization_id: int = Field(default=1, ge=1, description="Grafana organization ID")

    @field_validator("grafana_url")
    @classmethod
    def validate_grafana_url(cls, v: str | None) -> str | None:
        """Validate Grafana URL format."""
        if v is None:
            return v
        if not v.startswith(("http://", "https://")):
            msg = "Grafana URL must start with http:// or https://"
            raise ValueError(msg)
        return v.rstrip("/")


class AlertingConfig(BaseConfig):
    """Alerting configuration - consolidated pattern."""

    # Alert settings
    alerts_enabled: bool = Field(default=False, description="Enable alerting")
    alert_interval: int = Field(
        default=60,
        ge=30,
        le=3600,
        description="Alert evaluation interval in seconds",
    )
    repeat_interval: int = Field(
        default=300,
        ge=60,
        le=86400,
        description="Alert repeat interval in seconds",
    )

    # Notification channels
    email_enabled: bool = Field(default=False, description="Enable email notifications")
    slack_enabled: bool = Field(default=False, description="Enable Slack notifications")
    webhook_enabled: bool = Field(
        default=False,
        description="Enable webhook notifications",
    )

    # SMTP settings for email alerts
    smtp_host: str | None = Field(default=None, description="SMTP host")
    smtp_port: int = Field(default=587, ge=1, le=65535, description="SMTP port")
    smtp_username: str | None = Field(default=None, description="SMTP username")
    smtp_password: str | None = Field(
        default=None,
        description="SMTP password",
        json_schema_extra={"secret": True},
    )
    smtp_use_tls: bool = Field(default=True, description="Use TLS for SMTP")

    # Slack settings
    slack_webhook_url: str | None = Field(
        default=None,
        description="Slack webhook URL",
        json_schema_extra={"secret": True},
    )
    slack_channel: str | None = Field(default=None, description="Slack channel")

    # Webhook settings
    webhook_url: str | None = Field(default=None, description="Generic webhook URL")
    webhook_timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Webhook timeout in seconds",
    )

    @field_validator("slack_webhook_url", "webhook_url")
    @classmethod
    def validate_webhook_url(cls, v: str | None) -> str | None:
        """Validate webhook URL format."""
        if v is None:
            return v
        if not v.startswith("https://"):
            msg = "Webhook URL must use HTTPS"
            raise ValueError(msg)
        return v


class ObservabilityConfig(BaseConfig):
    """Complete observability configuration - aggregates all monitoring configs."""

    # Core monitoring
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig,
        description="Core monitoring settings",
    )

    # Metrics backends
    prometheus: PrometheusConfig = Field(
        default_factory=PrometheusConfig,
        description="Prometheus configuration",
    )
    grafana: GrafanaConfig = Field(
        default_factory=GrafanaConfig,
        description="Grafana configuration",
    )

    # Alerting
    alerting: AlertingConfig = Field(
        default_factory=AlertingConfig,
        description="Alerting configuration",
    )

    # Performance monitoring
    performance_monitoring: bool = Field(
        default=True,
        description="Enable performance monitoring",
    )
    error_tracking: bool = Field(default=True, description="Enable error tracking")
    request_tracing: bool = Field(default=False, description="Enable request tracing")

    # Resource monitoring
    cpu_monitoring: bool = Field(default=True, description="Enable CPU monitoring")
    memory_monitoring: bool = Field(
        default=True,
        description="Enable memory monitoring",
    )
    disk_monitoring: bool = Field(default=True, description="Enable disk monitoring")
    network_monitoring: bool = Field(
        default=True,
        description="Enable network monitoring",
    )


__all__ = [
    "AlertingConfig",
    "GrafanaConfig",
    "MonitoringConfig",
    "ObservabilityConfig",
    "PrometheusConfig",
]
