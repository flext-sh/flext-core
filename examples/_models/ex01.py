"""Example 01 models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import m

if TYPE_CHECKING:
    from examples.typings import t


class ExamplesFlextModelsEx01:
    """Example 01 models."""

    class User(m.Entity):
        """Result demo user model."""

        name: str = m.Field(description="User display name")
        email: str = m.Field(description="User email address")

    class DemonstrationResult(m.Value):
        """Result demo summary model."""

        demonstrations_completed: int = m.Field(
            description="Count of completed demonstrations"
        )
        patterns_covered: t.VariadicTuple[str] = m.Field(
            description="Tuple of covered pattern names"
        )
        completed_at: str = m.Field(description="ISO 8601 completion timestamp")

    class RunDemonstrationCommand(m.Command):
        """Result demo command model."""

        operation: str = m.Field(
            "demonstration", description="Operation type", validate_default=True
        )

    class ValidPersonPayload(m.Value):
        """Valid person payload model for demo validation."""

        name: str = m.Field(description="Person name")
        age: int = m.Field(description="Person age in years")

    class InvalidPersonPayload(m.Value):
        """Invalid person payload model (for failure path tests)."""

        name: str = m.Field(description="Person name")
        age: str = m.Field(
            description="Invalid age (string instead of int) for testing"
        )
