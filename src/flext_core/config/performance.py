"""Performance configuration patterns - consolidated from multiple projects.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module consolidates performance configuration patterns found across:
- flext-api, flext-oracle-wms, flext-tap-oracle, flext-grpc, flext-observability
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field
from pydantic import field_validator

from flext_core.config.base import BaseConfig

if TYPE_CHECKING:
    from pydantic import ValidationInfo


class PerformanceConfig(BaseConfig):
    """Performance configuration - consolidated pattern from 7+ projects."""

    # Connection and timeout settings
    timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Default timeout in seconds",
    )
    connection_timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Connection timeout in seconds",
    )
    request_timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Request timeout in seconds",
    )
    read_timeout: int = Field(
        default=60,
        ge=1,
        le=600,
        description="Read timeout in seconds",
    )

    # Retry settings
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts",
    )
    retry_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Retry delay in seconds",
    )
    retry_backoff_factor: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Retry backoff multiplier",
    )

    # Batch processing
    batch_size: int = Field(
        default=1000,
        ge=1,
        le=50000,
        description="Default batch size",
    )
    page_size: int = Field(default=500, ge=1, le=10000, description="Default page size")
    max_page_size: int = Field(
        default=5000,
        ge=1,
        le=50000,
        description="Maximum page size",
    )

    # Concurrency
    max_workers: int = Field(
        default=4,
        ge=1,
        le=50,
        description="Maximum worker threads",
    )
    max_parallel_streams: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Maximum parallel streams",
    )
    connection_pool_size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Connection pool size",
    )

    # Memory and resource limits
    max_memory_mb: int = Field(
        default=1024,
        ge=128,
        le=8192,
        description="Maximum memory usage in MB",
    )
    chunk_size: int = Field(
        default=8192,
        ge=1024,
        le=65536,
        description="Data chunk size in bytes",
    )

    # Feature flags
    enable_async: bool = Field(
        default=True,
        description="Enable asynchronous processing",
    )
    enable_caching: bool = Field(default=True, description="Enable caching")
    enable_compression: bool = Field(
        default=True,
        description="Enable data compression",
    )

    @field_validator("max_page_size")
    @classmethod
    def validate_max_page_size(cls, v: int, info: ValidationInfo) -> int:
        """Validate max_page_size is greater than page_size."""
        values = info.data if hasattr(info, "data") else {}
        page_size = values.get("page_size", 500)

        if v < page_size:
            msg = "max_page_size must be greater than or equal to page_size"
            raise ValueError(msg)
        return v


class DatabasePerformanceConfig(BaseConfig):
    """Database-specific performance configuration."""

    # Query optimization
    query_timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Query timeout in seconds",
    )
    max_query_complexity: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Maximum query complexity",
    )
    enable_query_cache: bool = Field(
        default=True,
        description="Enable query result caching",
    )
    cache_ttl: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Cache TTL in seconds",
    )

    # Connection pooling
    pool_size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Database connection pool size",
    )
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
        description="Pool checkout timeout",
    )
    pool_recycle: int = Field(
        default=3600,
        ge=0,
        description="Pool connection recycle time",
    )

    # Batch operations
    bulk_insert_size: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Bulk insert batch size",
    )
    bulk_update_size: int = Field(
        default=500,
        ge=1,
        le=5000,
        description="Bulk update batch size",
    )

    # Indexing and optimization
    auto_create_indexes: bool = Field(
        default=False,
        description="Automatically create indexes",
    )
    analyze_tables: bool = Field(default=True, description="Run table analysis")


class NetworkPerformanceConfig(BaseConfig):
    """Network-specific performance configuration."""

    # Connection settings
    keep_alive: bool = Field(default=True, description="Enable HTTP keep-alive")
    connection_pool_maxsize: int = Field(
        default=100,
        ge=1,
        le=500,
        description="HTTP connection pool max size",
    )
    max_retries_connect: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Connection retry attempts",
    )

    # Bandwidth and limits
    max_request_size_mb: int = Field(
        default=100,
        ge=1,
        le=1024,
        description="Maximum request size in MB",
    )
    max_response_size_mb: int = Field(
        default=500,
        ge=1,
        le=2048,
        description="Maximum response size in MB",
    )
    rate_limit_requests: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Rate limit requests per minute",
    )

    # Compression and encoding
    enable_gzip: bool = Field(default=True, description="Enable gzip compression")
    compression_level: int = Field(
        default=6,
        ge=1,
        le=9,
        description="Compression level (1-9)",
    )

    # SSL/TLS
    ssl_verify: bool = Field(default=True, description="Verify SSL certificates")
    ssl_timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="SSL handshake timeout",
    )


class CachePerformanceConfig(BaseConfig):
    """Cache-specific performance configuration."""

    # Cache settings
    default_ttl: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Default cache TTL in seconds",
    )
    max_cache_size_mb: int = Field(
        default=256,
        ge=1,
        le=2048,
        description="Maximum cache size in MB",
    )
    cache_key_prefix: str = Field(default="flext:", description="Cache key prefix")

    # Cache strategies
    enable_write_through: bool = Field(
        default=True,
        description="Enable write-through caching",
    )
    enable_write_behind: bool = Field(
        default=False,
        description="Enable write-behind caching",
    )
    cache_miss_penalty: float = Field(
        default=0.1,
        ge=0.0,
        le=10.0,
        description="Cache miss penalty factor",
    )

    # Eviction policies
    max_keys: int = Field(
        default=100000,
        ge=1000,
        le=1000000,
        description="Maximum number of cache keys",
    )
    eviction_policy: str = Field(default="lru", description="Cache eviction policy")

    @field_validator("eviction_policy")
    @classmethod
    def validate_eviction_policy(cls, v: str) -> str:
        """Validate cache eviction policy."""
        allowed_policies = {"lru", "lfu", "fifo", "random", "ttl"}

        if v.lower() not in allowed_policies:
            msg = f"Eviction policy must be one of: {allowed_policies}"
            raise ValueError(msg)
        return v.lower()


__all__ = [
    "CachePerformanceConfig",
    "DatabasePerformanceConfig",
    "NetworkPerformanceConfig",
    "PerformanceConfig",
]
