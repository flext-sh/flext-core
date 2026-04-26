"""Example 02 settings models."""

from __future__ import annotations

from flext_core import c, m, p, r, u


class ExamplesFlextCoreModelsEx02:
    """Example 02 model namespace."""

    class DatabaseService(m.Value):
        """Database service model used in example 02 settings integration."""

        settings: m.ConfigMap = u.Field(description="Database connection settings")
        status: c.Status = u.Field(
            c.Status.PENDING,
            description="Service connection status",
            validate_default=True,
        )

        def connect(self) -> p.Result[bool]:
            return r[bool].ok(True)

        def query(self, sql: str) -> p.Result[m.ConfigMap]:
            if "INVALID" in sql:
                return r[m.ConfigMap].fail("invalid query")
            return r[m.ConfigMap].ok(m.ConfigMap(root={"rows": 1}))

    class CacheService(m.Value):
        """Cache service model used in example 02 settings integration."""

        settings: m.ConfigMap = u.Field(description="Cache connection settings")
        status: c.Status = u.Field(
            c.Status.PENDING,
            description="Service connection status",
            validate_default=True,
        )

        def set(self, key: str, value: str) -> p.Result[bool]:
            if not key:
                return r[bool].fail("missing key")
            if not value:
                return r[bool].fail("missing value")
            return r[bool].ok(True)

    class EmailService(m.Value):
        """Email service model used in example 02 settings integration."""

        settings: m.ConfigMap = u.Field(description="Email service settings")
        status: c.Status = u.Field(
            c.Status.PENDING,
            description="Service connection status",
            validate_default=True,
        )

        def send(self, to: str, subject: str, body: str) -> p.Result[bool]:
            if not to or not subject or (not body):
                return r[bool].fail("invalid email payload")
            return r[bool].ok(True)
