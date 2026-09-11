"""Extracted mixin for flext_tests."""

from __future__ import annotations

from collections.abc import Callable


class FlextTestsContainerHelpersUtilitiesMixin:
    """Helpers for container testing."""

    @staticmethod
    def create_counting_factory[TFactory](
        return_value: TFactory,
    ) -> tuple[Callable[[], TFactory], Callable[[], int]]:
        """Create a factory that counts invocations.

        Args:
            return_value: Value to return from factory

        Returns:
            r[TEntity]: Result containing created entity or error
            Tuple of (factory function, count getter)

        """
        count = [0]

        def factory() -> TFactory:
            count[0] += 1
            return return_value

        def count_value() -> int:
            return count[0]

        return (factory, count_value)

    @staticmethod
    def create_factory[TFactory](return_value: TFactory) -> Callable[[], TFactory]:
        """Create a factory function that returns a fixed value.

        Args:
            return_value: Value to return from factory

        Returns:
            r[TEntity]: Result containing created entity or error
            Factory function

        """

        def factory() -> TFactory:
            return return_value

        return factory
