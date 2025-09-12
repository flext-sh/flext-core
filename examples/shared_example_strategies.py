"""Shared example strategies for FLEXT examples.

This module provides common strategies and factories used across examples.
"""


from flext_core.types import FlextTypes


class DemoStrategy:
    """Demo strategy for examples."""

    def execute(self, data: FlextTypes.Core.Dict) -> FlextTypes.Core.Dict:
        """Execute strategy."""
        return data


class ExamplePatternFactory:
    """Factory for example patterns."""

    @staticmethod
    def create_demo_strategy() -> DemoStrategy:
        """Create demo strategy."""
        return DemoStrategy()

    @staticmethod
    def create_demo_runner() -> DemoStrategy:
        """Create demo runner."""
        return DemoStrategy()

    @staticmethod
    def create_pattern(name: str) -> DemoStrategy | None:
        """Create pattern by name."""
        if name == "demo":
            return DemoStrategy()
        return None
