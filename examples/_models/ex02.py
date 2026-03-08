"""Example 02 settings models."""

from __future__ import annotations

from pydantic import Field

from flext_core import FlextSettings, c, m, r


class Ex02TestConfig(FlextSettings):
    """Settings model used by Ex02 settings golden tests."""

    service_name: str = Field(default="example-service")
    feature_enabled: bool = Field(default=True)


class Ex02DatabaseService(m.Value):
    config: m.ConfigMap
    status: c.Cqrs.CommonStatus = c.Cqrs.CommonStatus.PENDING

    def connect(self) -> r[bool]:
        return r[bool].ok(True)

    def query(self, sql: str) -> r[m.ConfigMap]:
        if "INVALID" in sql:
            return r[m.ConfigMap].fail("invalid query")
        return r[m.ConfigMap].ok(m.ConfigMap(root={"rows": 1}))


class Ex02CacheService(m.Value):
    config: m.ConfigMap
    status: c.Cqrs.CommonStatus = c.Cqrs.CommonStatus.PENDING

    def set(self, key: str, value: str) -> r[bool]:
        if not key:
            return r[bool].fail("missing key")
        if not value:
            return r[bool].fail("missing value")
        return r[bool].ok(True)


class Ex02EmailService(m.Value):
    config: m.ConfigMap
    status: c.Cqrs.CommonStatus = c.Cqrs.CommonStatus.PENDING

    def send(self, to: str, subject: str, body: str) -> r[bool]:
        if not to or not subject or not body:
            return r[bool].fail("invalid email payload")
        return r[bool].ok(True)
