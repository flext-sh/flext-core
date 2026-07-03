"""FlextContext — scoped key-value store + contextvar facade.

Pure Pydantic v2 model (m.ManagedModel). Zero utility-chain inheritance.
Context-variable I/O (correlation IDs, service metadata) exclusively via u.*.
Auto-injected transparently by FlextService via __init_subclass__.

Per AGENTS.md §0.7: zero nested classes; all methods flat on the facade.
Per AGENTS.md §3.1: single concern — context data model + contextvar ops.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timedelta

from flext_core import c, m, t, u

from .flextcontext_part_01 import (
    FlextContext as FlextContextPart01,
)


class FlextContext(FlextContextPart01):
    @staticmethod
    @contextmanager
    def new_correlation(
        correlation_id: str | None = None,
        parent_id: str | None = None,
    ) -> Generator[str]:
        """Scope a correlation ID, restoring the previous one on exit."""
        if correlation_id is None:
            correlation_id = u.generate("correlation")
        current = u.CORRELATION_ID.get()
        corr_token = u.CORRELATION_ID.set(correlation_id)
        parent_token = (
            u.PARENT_CORRELATION_ID.set(parent_id)
            if parent_id
            else (
                u.PARENT_CORRELATION_ID.set(current)
                if isinstance(current, str)
                else None
            )
        )
        try:
            yield correlation_id
        finally:
            u.CORRELATION_ID.reset(corr_token)
            if parent_token:
                u.PARENT_CORRELATION_ID.reset(parent_token)

    @staticmethod
    def apply_correlation_id(correlation_id: str | None) -> None:
        """Set correlation ID in process context."""
        _ = u.CORRELATION_ID.set(correlation_id)

    @staticmethod
    def ensure_correlation_id() -> str:
        """Return current correlation ID, generating one if absent."""
        current = u.CORRELATION_ID.get()
        if isinstance(current, str) and current:
            return current
        new_id: str = u.generate("correlation")
        _ = u.CORRELATION_ID.set(new_id)
        return new_id

    @staticmethod
    @contextmanager
    def service_context(
        service_name: str,
        version: str | None = None,
    ) -> Generator[None]:
        """Scope service name/version in process context."""
        name_token = u.SERVICE_NAME.set(service_name)
        version_token = u.SERVICE_VERSION.set(version) if version else None
        try:
            yield
        finally:
            u.SERVICE_NAME.reset(name_token)
            if version_token:
                u.SERVICE_VERSION.reset(version_token)

    @staticmethod
    def resolve_operation_name() -> str | None:
        """Get current operation name from process context."""
        v = u.OPERATION_NAME.get()
        return str(v) if v is not None else None

    @staticmethod
    def apply_operation_name(operation_name: str) -> None:
        """Set operation name in process context."""
        _ = u.OPERATION_NAME.set(operation_name)

    @staticmethod
    @contextmanager
    def timed_operation(
        operation_name: str | None = None,
    ) -> Generator[m.ConfigMap]:
        """Scope a timed operation with performance metadata."""
        start_time = u.generate_datetime_utc()
        start_perf = time.perf_counter()
        payload = t.json_mapping_adapter().validate_python({
            str(c.MetadataKey.START_TIME): start_time.isoformat(),
            str(c.ContextKey.OPERATION_NAME): operation_name or "",
        })
        op_meta = m.ConfigMap.model_validate(payload)
        start_token = u.OPERATION_START_TIME.set(start_time)
        meta_token = u.OPERATION_METADATA.set(payload)
        op_token = u.OPERATION_NAME.set(operation_name) if operation_name else None
        try:
            yield op_meta
        finally:
            duration = time.perf_counter() - start_perf
            end_time = start_time + timedelta(seconds=duration)
            op_meta.update({
                c.MetadataKey.END_TIME: end_time.isoformat(),
                c.MetadataKey.DURATION_SECONDS: duration,
            })
            u.OPERATION_START_TIME.reset(start_token)
            u.OPERATION_METADATA.reset(meta_token)
            if op_token:
                u.OPERATION_NAME.reset(op_token)

    @staticmethod
    def export_full_context() -> t.MappingKV[str, t.Scalar]:
        """Export all active contextvar values as a flat mapping."""
        result: dict[str, t.Scalar] = {}
        if (v := u.CORRELATION_ID.get()) is not None:
            result[c.ContextKey.CORRELATION_ID] = str(v)
        if (v := u.PARENT_CORRELATION_ID.get()) is not None:
            result[c.ContextKey.PARENT_CORRELATION_ID] = str(v)
        if (v := u.SERVICE_NAME.get()) is not None:
            result[c.ContextKey.SERVICE_NAME] = str(v)
        if (v := u.SERVICE_VERSION.get()) is not None:
            result[c.ContextKey.SERVICE_VERSION] = str(v)
        if (v := u.USER_ID.get()) is not None:
            result[c.ContextKey.USER_ID] = str(v)
        if (v := u.REQUEST_ID.get()) is not None:
            result[c.ContextKey.REQUEST_ID] = str(v)
        if (v := u.OPERATION_NAME.get()) is not None:
            result[c.ContextKey.OPERATION_NAME] = str(v)
        if (v := u.OPERATION_START_TIME.get()) is not None:
            result[c.ContextKey.OPERATION_START_TIME] = (
                v.isoformat() if isinstance(v, datetime) else str(v)
            )
        return result

    @staticmethod
    def clear_context() -> None:
        """Clear all contextvar proxies (process-global scope)."""
        for ctx_var in (
            u.CORRELATION_ID,
            u.PARENT_CORRELATION_ID,
            u.SERVICE_NAME,
            u.SERVICE_VERSION,
            u.USER_ID,
            u.REQUEST_ID,
            u.OPERATION_NAME,
        ):
            _ = ctx_var.set(None)
        _ = u.OPERATION_START_TIME.set(None)
        _ = u.OPERATION_METADATA.set(None)
        _ = u.REQUEST_TIMESTAMP.set(None)


__all__: list[str] = ["FlextContext"]
