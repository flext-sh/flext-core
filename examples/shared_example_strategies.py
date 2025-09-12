"""Shared example strategies for FLEXT examples.

This module provides common strategies and factories used across examples.
"""

from flext_core import FlextResult, FlextTypes


class DemoStrategy:
    """Demo strategy for examples."""

    def execute(
        self, data: FlextTypes.Core.Dict | None = None
    ) -> FlextResult[FlextTypes.Core.Dict]:
        """Execute strategy."""
        # Use data parameter to avoid unused argument warning
        result_data: FlextTypes.Core.Dict = {
            "status": "success",
            "message": "Demo executed",
        }
        if data:
            result_data["input_data"] = data
        return FlextResult[FlextTypes.Core.Dict].ok(result_data)


class ExamplePatternFactory:
    """Factory for example patterns."""

    @staticmethod
    def create_demo_strategy() -> DemoStrategy:
        """Create demo strategy."""
        return DemoStrategy()

    @staticmethod
    def create_demo_runner(
        name: str | None = None, func: object = None
    ) -> DemoStrategy:
        """Create demo runner."""
        # Parameters are reserved for future use but kept for API compatibility
        _ = (name, func)  # Mark as intentionally unused
        return DemoStrategy()

    @staticmethod
    def create_pattern(name: str) -> DemoStrategy | None:
        """Create pattern by name."""
        if name == "demo":
            return DemoStrategy()
        return None
