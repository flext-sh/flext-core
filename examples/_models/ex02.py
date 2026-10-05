"""Example 02 settings models.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import c, m, p, r


class ExamplesFlextModelsEx02:
    """Example 02 model namespace."""

    class DatabaseService(m.Value):
        """Database service model used in example 02 settings integration."""

        settings: m.ConfigMap = m.Field(description="Database connection settings")
        status: c.Status = m.Field(
            c.Status.PENDING,
            description="Service connection status",
            validate_default=True,
        )

        @staticmethod
        def connect() -> p.Result[bool]:
            return r[bool].ok(True)

        @staticmethod
        def query(sql: str) -> p.Result[m.ConfigMap]:
            if "INVALID" in sql:
                return r[m.ConfigMap].fail("invalid query")
            return r[m.ConfigMap].ok(m.ConfigMap(root={"rows": 1}))

    class CacheService(m.Value):
        """Cache service model used in example 02 settings integration."""

        settings: m.ConfigMap = m.Field(description="Cache connection settings")
        status: c.Status = m.Field(
            c.Status.PENDING,
            description="Service connection status",
            validate_default=True,
        )

        @staticmethod
        def set(key: str, value: str) -> p.Result[bool]:
            if not key:
                return r[bool].fail("missing key")
            if not value:
                return r[bool].fail("missing value")
            return r[bool].ok(True)

    class EmailService(m.Value):
        """Email service model used in example 02 settings integration."""

        settings: m.ConfigMap = m.Field(description="Email service settings")
        status: c.Status = m.Field(
            c.Status.PENDING,
            description="Service connection status",
            validate_default=True,
        )

        @staticmethod
        def send(to: str, subject: str, body: str) -> p.Result[bool]:
            if not to or not subject or (not body):
                return r[bool].fail("invalid email payload")
            return r[bool].ok(True)
