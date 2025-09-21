"""Shared Strategy Pattern for Examples - Eliminates Code Duplication.

This module provides centralized strategies and factories to eliminate code
duplication across all examples while maintaining backward compatibility.

Uses Factory Pattern, Strategy Pattern, and Template Method Pattern from flext-core.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from functools import reduce
from typing import NamedTuple, Protocol, TypeVar

from flext_core import FlextLogger, FlextResult

T = TypeVar("T")

logger = FlextLogger(__name__)


class DemoStrategy(Protocol[T]):
    """Strategy interface for demonstration patterns using flext-core protocols."""

    def execute(self) -> FlextResult[T]:
        """Execute demonstration strategy."""
        ...

    def cleanup(self) -> FlextResult[None]:
        """Clean up resources after demonstration."""
        ...


class ExamplePatternFactory:
    """Factory Pattern for creating example demonstrations using flext-core patterns.

    Eliminates code duplication by centralizing common example execution logic
    following Factory Pattern and using FlextResult for error handling.
    """

    @staticmethod
    def create_demo_runner[T](
        name: str,
        demo_func: Callable[[], FlextResult[T]],
        cleanup_func: Callable[[], FlextResult[None]] | None = None,
    ) -> DemoStrategy[T]:
        """Create a demo runner strategy using Factory Pattern.

        Args:
            name: Name of the demonstration
            demo_func: Function that executes the demonstration
            cleanup_func: Optional cleanup function

        Returns:
            DemoStrategy[T]: Demo runner strategy instance

        """

        class DemoRunner:
            def __init__(self) -> None:
                self.name = name
                self.demo_func = demo_func
                self.cleanup_func = cleanup_func

            def execute(self) -> FlextResult[T]:
                """Execute demo with standardized error handling using Railway Pattern.

                Returns:
                    FlextResult[T]: Demo execution result

                """
                try:
                    logger.info(f"🚀 Starting demonstration: {self.name}")
                    result = self.demo_func()

                    if result.is_success:
                        logger.info(f"✅ {self.name} completed successfully")
                    else:
                        logger.error(f"❌ {self.name} failed: {result.error}")

                    return result

                except Exception as e:
                    error_msg = (
                        f"Demonstration '{self.name}' failed with exception: {e}"
                    )
                    logger.exception(error_msg)
                    return FlextResult[T].fail(error_msg)

            def cleanup(self) -> FlextResult[None]:
                """Standardized cleanup using Railway Pattern.

                Returns:
                    FlextResult[None]: Cleanup result

                """
                if self.cleanup_func:
                    try:
                        return self.cleanup_func()
                    except Exception as e:
                        logger.warning(f"Cleanup failed for {self.name}: {e}")
                        return FlextResult[None].fail(f"Cleanup failed: {e}")
                return FlextResult[None].ok(None)

        return DemoRunner()

    @staticmethod
    def create_validation_demo(
        name: str,
        data: dict[str, object],
        validation_rules: list[Callable[[dict[str, object]], FlextResult[None]]],
    ) -> DemoStrategy[dict[str, object]]:
        """Create validation demonstration using Railway Pattern - ELIMINATED LOOP RETURNS.

        Args:
            name: Name of the validation demo
            data: Data to validate
            validation_rules: List of validation rule functions

        Returns:
            DemoStrategy[dict[str, object]]: Validation demo strategy

        """

        def validation_demo() -> FlextResult[dict[str, object]]:
            """Execute validation demonstration with Functional Railway Pattern.

            Returns:
                FlextResult[dict[str, object]]: Validation result

            """
            logger.info(f"📋 Validating data for {name}")

            # Railway Pattern: Use reduce for functional validation chain

            def _apply_validation(
                acc_result: FlextResult[dict[str, object]],
                rule: Callable[[dict[str, object]], FlextResult[None]],
            ) -> FlextResult[dict[str, object]]:
                """Apply single validation rule in railway chain.
                
                Args:
                    acc_result: Accumulated validation result
                    rule: Validation rule to apply
                    
                Returns:
                    FlextResult[dict[str, object]]: Updated validation result
                """
                return acc_result.flat_map(
                    lambda data_dict: rule(data_dict).map(lambda _: data_dict),
                ).tap_error(lambda e: logger.error(f"Validation failed: {e}"))

            # Functional approach: Chain all validations without loop returns
            initial_result = FlextResult[dict[str, object]].ok(data)
            final_result = reduce(_apply_validation, validation_rules, initial_result)

            if final_result.is_success:
                logger.info(f"✅ All validations passed for {name}")

            return final_result

        return ExamplePatternFactory.create_demo_runner(name, validation_demo)

    @staticmethod
    def create_configuration_demo(
        name: str,
        config_class: type,
        config_data: dict[str, object],
    ) -> DemoStrategy[object]:
        """Create configuration demonstration using Railway Pattern - ELIMINATED TRY/CATCH RETURNS."""

        def config_demo() -> FlextResult[object]:
            """Execute configuration demonstration with Pure Railway Pattern."""
            logger.info(f"⚙️ Creating configuration for {name}")

            # Railway Pattern: Safe configuration creation
            return (
                _safe_create_config_instance(config_class, config_data)
                .flat_map(_validate_config_if_needed)
                .map(lambda config: _log_success_and_return(config, name))
                .tap_error(
                    lambda e: logger.error(f"Configuration creation failed: {e}"),
                )
            )

        # Railway Helper Functions
        def _safe_create_config_instance(
            config_class: type,
            config_data: dict[str, object],
        ) -> FlextResult[object]:
            """Safely create config instance with error handling."""
            try:
                config_instance = config_class(**config_data)
                return FlextResult[object].ok(config_instance)
            except Exception as e:
                return FlextResult[object].fail(str(e))

        def _validate_config_if_needed(config_instance: object) -> FlextResult[object]:
            """Conditionally validate config using Railway Pattern."""
            if not hasattr(config_instance, "validate_business_rules"):
                return FlextResult[object].ok(config_instance)

            # Check if config_instance has validate_business_rules method
            validation_result = getattr(
                config_instance,
                "validate_business_rules",
                lambda: FlextResult[None].ok(None),
            )()
            return validation_result.flat_map(
                lambda _: FlextResult[object].ok(config_instance),
            ).tap_error(lambda e: logger.error(f"Config validation failed: {e}"))

        def _log_success_and_return(config_instance: object, name: str) -> object:
            """Log success and return config instance."""
            logger.info(f"✅ Configuration created successfully for {name}")
            return config_instance

        return ExamplePatternFactory.create_demo_runner(name, config_demo)

    @staticmethod
    def execute_demo_pipeline(
        demos: list[DemoStrategy[object]],
    ) -> FlextResult[list[object]]:
        """Execute multiple demonstrations in pipeline using Railway Pattern - ELIMINATED LOOP RETURNS."""
        logger.info(f"🔄 Starting demo pipeline with {len(demos)} demonstrations")

        # Railway Pattern: Use functional approach with fold/reduce
        class PipelineState(NamedTuple):
            results: list[object]
            executed_demos: list[DemoStrategy[object]]

        def _execute_single_demo(
            acc_state: FlextResult[PipelineState],
            demo: DemoStrategy[object],
        ) -> FlextResult[PipelineState]:
            """Execute single demo with error propagation using Railway Pattern."""
            return acc_state.flat_map(
                lambda state: demo.execute()
                .map(
                    lambda result: PipelineState(
                        results=[*state.results, result],
                        executed_demos=[*state.executed_demos, demo],
                    ),
                )
                .tap_error(lambda e: _cleanup_on_failure(state.executed_demos, e)),
            )

        def _cleanup_on_failure(
            executed_demos: list[DemoStrategy[object]],
            error: str,
        ) -> None:
            """Cleanup executed demos on failure."""
            logger.error(f"❌ Pipeline failed: {error}")
            for cleanup_demo in executed_demos:
                cleanup_demo.cleanup()

        def _perform_final_cleanup(final_state: PipelineState) -> list[object]:
            """Perform cleanup on all demos and return results."""
            # Cleanup all demos after successful execution
            for demo in final_state.executed_demos:
                cleanup_result = demo.cleanup()
                if cleanup_result.is_failure:
                    logger.warning(f"⚠️ Cleanup warning: {cleanup_result.error}")

            logger.info("🎉 Demo pipeline completed successfully")
            return final_state.results

        # Railway Pattern execution: Chain all demos functionally
        initial_state = FlextResult[PipelineState].ok(
            PipelineState(results=[], executed_demos=[]),
        )

        return reduce(_execute_single_demo, demos, initial_state).map(
            _perform_final_cleanup,
        )

    @staticmethod
    def create_composite_demo_suite(
        suite_name: str,
        demos: list[tuple[str, Callable[[], FlextResult[object]]]],
    ) -> FlextResult[str]:
        """Create composite demo suite to eliminate duplication - ANTI-DUPLICATION PATTERN.

        This method consolidates common demo execution patterns found in multiple examples,
        reducing the 26-line duplication (mass=172) between handlers and exceptions examples.
        """
        logger.info(f"🎯 Executing {suite_name} Demo Suite")

        success_count = 0
        total_demos = len(demos)
        results = []

        for demo_name, demo_func in demos:
            demo_runner = ExamplePatternFactory.create_demo_runner(
                demo_name,
                demo_func,
                lambda: FlextResult[None].ok(None),  # Standard cleanup
            )

            result = demo_runner.execute()
            if result.is_success:
                success_count += 1
                results.append(f"✅ {demo_name}: Success")
            else:
                results.append(f"❌ {demo_name}: {result.error}")

        success_rate = (success_count / total_demos) * 100.0

        summary = (
            f"{suite_name} Suite Results:\n"
            f"Total: {total_demos}, Success: {success_count}, Rate: {success_rate:.1f}%\n"
            + "\n".join(results)
        )

        if success_rate >= 80.0:
            logger.info(
                f"✅ {suite_name} suite completed with {success_rate:.1f}% success",
            )
            return FlextResult[str].ok(summary)
        logger.warning(
            f"⚠️ {suite_name} suite completed with suboptimal {success_rate:.1f}% success",
        )
        return FlextResult[str].fail(f"Suboptimal execution: {summary}")


def main() -> None:
    """Demonstrate shared example strategies and patterns.

    This function showcases the capabilities of the ExamplePatternFactory
    and demonstrates the various strategy patterns available.
    """
    logger.info("🎯 Demonstrating Shared Example Strategies")

    # Demonstrate simple demo runner
    simple_demo = ExamplePatternFactory.create_demo_runner(
        "Simple Demo",
        lambda: FlextResult[str].ok("Demo executed successfully"),
    )

    result = simple_demo.execute()
    if result.is_success:
        logger.info(f"Demo result: {result.unwrap()}")

    # Demonstrate validation demo
    sample_data = {"name": "test", "value": 42}
    validation_rules = [
        lambda data: FlextResult[None].ok(None)
        if "name" in data
        else FlextResult[None].fail("Missing name"),
        lambda data: FlextResult[None].ok(None)
        if isinstance(data.get("value"), int)
        else FlextResult[None].fail("Invalid value type"),
    ]

    validation_demo = ExamplePatternFactory.create_validation_demo(
        "Data Validation",
        sample_data,
        validation_rules,
    )

    validation_result = validation_demo.execute()
    if validation_result.is_success:
        logger.info("Validation demo completed successfully")

    logger.info("✅ All strategy demonstrations completed")


__all__ = [
    "DemoStrategy",
    "ExamplePatternFactory",
]


if __name__ == "__main__":
    main()
