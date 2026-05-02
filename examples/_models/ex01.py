"""Example 01 models."""

from __future__ import annotations

from examples import t

from flext_core import m, u


class ExamplesFlextModelsEx01:
    """Example 01 models."""

    class User(m.Entity):
        """Result demo user model."""

        name: str = u.Field(description="User display name")
        email: str = u.Field(description="User email address")

    class DemonstrationResult(m.Value):
        """Result demo summary model."""

        demonstrations_completed: int = u.Field(
            description="Count of completed demonstrations"
        )
        patterns_covered: t.VariadicTuple[str] = u.Field(
            description="Tuple of covered pattern names"
        )
        completed_at: str = u.Field(description="ISO 8601 completion timestamp")

    class RunDemonstrationCommand(m.Command):
        """Result demo command model."""

        operation: str = u.Field(
            "demonstration",
            description="Operation type",
            validate_default=True,
        )

    class ValidPersonPayload(m.Value):
        """Valid person payload model for demo validation."""

        name: str = u.Field(description="Person name")
        age: int = u.Field(description="Person age in years")

    class InvalidPersonPayload(m.Value):
        """Invalid person payload model (for failure path tests)."""

        name: str = u.Field(description="Person name")
        age: str = u.Field(
            description="Invalid age (string instead of int) for testing"
        )
