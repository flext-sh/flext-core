"""Database configuration patterns - consolidated from multiple projects.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module consolidates database configuration patterns found across:
- flext-api, flext-auth, flext-cli, flext-grpc, flext-observability, flext-web
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field
from pydantic import field_validator

from flext_core.config.base import BaseConfig

if TYPE_CHECKING:
    from pydantic import ValidationInfo


class DatabaseConfig(BaseConfig):
    """Database connection configuration - consolidated pattern."""

    url: str = Field(..., description="Database connection URL")
    pool_size: int = Field(default=20, ge=1, le=100, description="Connection pool size")
    max_overflow: int = Field(
        default=40,
        ge=0,
        le=200,
        description="Maximum pool overflow",
    )
    pool_timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Pool timeout in seconds",
    )
    pool_recycle: int = Field(
        default=3600,
        ge=0,
        description="Pool recycle time in seconds",
    )
    echo: bool = Field(default=False, description="Enable SQL query logging")
    echo_pool: bool = Field(default=False, description="Enable connection pool logging")

    @property
    def async_url(self) -> str:
        """Convert sync URL to async URL for asyncpg driver."""
        if self.url.startswith("postgresql://"):
            return self.url.replace("postgresql://", "postgresql+asyncpg://")
        if self.url.startswith("sqlite://"):
            return self.url.replace("sqlite://", "sqlite+aiosqlite://")
        return self.url

    @field_validator("url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Validate database URL format."""
        if not any(
            v.startswith(scheme)
            for scheme in [
                "postgresql://",
                "mysql://",
                "sqlite://",
                "oracle://",
                "postgresql+asyncpg://",
                "mysql+aiomysql://",
                "sqlite+aiosqlite://",
            ]
        ):
            msg = (
                "Database URL must start with supported scheme: "
                "postgresql://, mysql://, sqlite://, oracle://, "
                "postgresql+asyncpg://, mysql+aiomysql://, sqlite+aiosqlite://"
            )
            raise ValueError(
                msg,
            )
        return v


class PostgreSQLConfig(DatabaseConfig):
    """PostgreSQL-specific database configuration."""

    url: str = Field(
        default="postgresql://localhost/flext",
        description="PostgreSQL connection URL",
    )

    @field_validator("url")
    @classmethod
    def validate_postgresql_url(cls, v: str) -> str:
        """Validate PostgreSQL URL format."""
        if not v.startswith(("postgresql://", "postgresql+asyncpg://")):
            msg = (
                "PostgreSQL URL must start with postgresql:// or postgresql+asyncpg://"
            )
            raise ValueError(
                msg,
            )
        return v


class SQLiteConfig(DatabaseConfig):
    """SQLite-specific database configuration."""

    url: str = Field(
        default="sqlite:///./flext.db",
        description="SQLite connection URL",
    )
    pool_size: int = Field(
        default=1,
        ge=1,
        le=1,
        description="SQLite uses single connection",
    )
    max_overflow: int = Field(
        default=0,
        ge=0,
        le=0,
        description="SQLite doesn't support overflow",
    )

    @field_validator("url")
    @classmethod
    def validate_sqlite_url(cls, v: str) -> str:
        """Validate SQLite URL format."""
        if not v.startswith(("sqlite://", "sqlite+aiosqlite://")):
            msg = "SQLite URL must start with sqlite:// or sqlite+aiosqlite://"
            raise ValueError(
                msg,
            )
        return v


class OracleConfig(BaseConfig):
    """Oracle database configuration - consolidated from Oracle projects."""

    host: str = Field(..., description="Oracle database host")
    port: int = Field(default=1521, ge=1, le=65535, description="Oracle database port")
    service_name: str | None = Field(default=None, description="Oracle service name")
    sid: str | None = Field(default=None, description="Oracle SID")
    username: str = Field(..., description="Oracle username")
    password: str = Field(
        ...,
        description="Oracle password",
        json_schema_extra={"secret": True},
    )

    # Connection options
    encoding: str = Field(default="UTF-8", description="Character encoding")
    nencoding: str = Field(default="UTF-8", description="National character encoding")
    threaded: bool = Field(default=True, description="Enable threaded mode")

    # Connection pooling
    min_pool_size: int = Field(default=1, ge=1, le=50, description="Minimum pool size")
    max_pool_size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum pool size",
    )
    increment: int = Field(default=1, ge=1, le=10, description="Pool increment")

    @field_validator("service_name", "sid")
    @classmethod
    def validate_service_or_sid(cls, v: str | None, info: ValidationInfo) -> str | None:
        """Validate that either service_name or sid is provided."""
        values = info.data if hasattr(info, "data") else {}
        if not values.get("service_name") and not values.get("sid"):
            msg = "Either service_name or sid must be provided"
            raise ValueError(msg)
        return v

    @property
    def connection_string(self) -> str:
        """Build Oracle connection string."""
        if self.service_name:
            return f"{self.host}:{self.port}/{self.service_name}"
        if self.sid:
            return f"{self.host}:{self.port}:{self.sid}"
        msg = "Either service_name or sid must be provided"
        raise ValueError(msg)

    @property
    def dsn(self) -> str:
        """Build Oracle DSN string."""
        return self.connection_string


__all__ = [
    "DatabaseConfig",
    "OracleConfig",
    "PostgreSQLConfig",
    "SQLiteConfig",
]
