"""Example 02 settings models."""

from __future__ import annotations

from pydantic import Field

from examples import c, t
from flext_core import FlextSettings, m, r


class Ex02TestConfig(FlextSettings):
    """Settings model used by Ex02 settings golden tests."""

    service_name: str = Field(default="example-service")
    feature_enabled: bool = Field(default=True)


class Ex02DatabaseService(m.Value):
    """Database service model used in example 02 settings integration."""

    settings: t.ConfigMap
    status: c.CommonStatus = c.CommonStatus.PENDING

    def connect(self) -> r[bool]:
        return r[bool].ok(True)

    def query(self, sql: str) -> r[t.ConfigMap]:
        if "INVALID" in sql:
            return r[t.ConfigMap].fail("invalid query")
        return r[t.ConfigMap].ok(t.ConfigMap(root={"rows": 1}))


class Ex02CacheService(m.Value):
    """Cache service model used in example 02 settings integration."""

    settings: t.ConfigMap
    status: c.CommonStatus = c.CommonStatus.PENDING

    def set(self, key: str, value: str) -> r[bool]:
        if not key:
            return r[bool].fail("missing key")
        if not value:
            return r[bool].fail("missing value")
        return r[bool].ok(True)


class Ex02EmailService(m.Value):
    """Email service model used in example 02 settings integration."""

    settings: t.ConfigMap
    status: c.CommonStatus = c.CommonStatus.PENDING

    def send(self, to: str, subject: str, body: str) -> r[bool]:
        if not to or not subject or (not body):
            return r[bool].fail("invalid email payload")
        return r[bool].ok(True)
