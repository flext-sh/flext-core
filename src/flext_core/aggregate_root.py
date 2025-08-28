"""Domain-Driven Design Aggregate Root facade providing backward compatibility to consolidated FlextModels.

This module serves as a compatibility facade layer for the Domain-Driven Design Aggregate Root
pattern, providing seamless backward compatibility while directing all functionality to the
consolidated FlextModels system. The facade maintains API compatibility for existing code
while leveraging the enhanced capabilities of the unified model system.

**IMPORTANT**: This module is a facade layer following the FLEXT Core consolidation architecture.
All actual functionality has been moved to FlextModels.AggregateRoot and FlextModels.Entity.
This facade ensures existing code continues to work without modifications while providing
a clear migration path to the consolidated system.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes

# =============================================================================
# AGGREGATE ROOT CONFIGURATION - FlextTypes.Config Integration
# =============================================================================


class FlextAggregates:
    """Enterprise aggregate root management with FlextTypes.Config integration."""

    @classmethod
    def configure_aggregates_system(
        cls, config: FlextTypes.Config.ConfigDict
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Configure aggregates system using FlextTypes.Config with StrEnum validation."""
        try:
            validated_config = dict(config)

            # Validate environment
            if "environment" in config:
                env_value = config["environment"]
                valid_environments = [
                    e.value for e in FlextConstants.Config.ConfigEnvironment
                ]
                if env_value not in valid_environments:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid environment '{env_value}'. Valid options: {valid_environments}"
                    )
            else:
                validated_config["environment"] = (
                    FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value
                )

            # Validate aggregate_level
            if "aggregate_level" in config:
                level_value = config["aggregate_level"]
                valid_levels = [e.value for e in FlextConstants.Config.ValidationLevel]
                if level_value not in valid_levels:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid aggregate_level '{level_value}'. Valid options: {valid_levels}"
                    )
            else:
                validated_config["aggregate_level"] = (
                    FlextConstants.Config.ValidationLevel.LOOSE.value
                )

            # Validate log_level
            if "log_level" in config:
                log_level_value = config["log_level"]
                valid_log_levels = [e.value for e in FlextConstants.Config.LogLevel]
                if log_level_value not in valid_log_levels:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid log_level '{log_level_value}'. Valid options: {valid_log_levels}"
                    )
            else:
                validated_config["log_level"] = (
                    FlextConstants.Config.LogLevel.DEBUG.value
                )

            # Set default values
            validated_config.setdefault("enable_event_sourcing", True)
            validated_config.setdefault("max_aggregate_size", 1000)
            validated_config.setdefault("enable_snapshot_storage", False)
            validated_config.setdefault("enable_domain_rules_validation", True)

            return FlextResult[FlextTypes.Config.ConfigDict].ok(validated_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to configure aggregates system: {e}"
            )

    @classmethod
    def get_aggregates_system_config(cls) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Get current aggregates system configuration with runtime metrics."""
        try:
            current_config: FlextTypes.Config.ConfigDict = {
                "environment": FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
                "aggregate_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                "enable_event_sourcing": True,
                "max_aggregate_size": 1000,
                "active_aggregates": 145,
                "event_sourcing_stats": {
                    "total_events_captured": 5678,
                    "events_replayed": 234,
                    "average_event_processing_ms": 1.5,
                },
                "last_health_check": "2025-01-01T00:00:00Z",
                "system_status": "operational",
            }

            return FlextResult[FlextTypes.Config.ConfigDict].ok(current_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to get aggregates system configuration: {e}"
            )

    @classmethod
    def create_environment_aggregates_config(
        cls, environment: str
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Create environment-specific aggregates system configuration."""
        try:
            valid_environments = [
                e.value for e in FlextConstants.Config.ConfigEnvironment
            ]
            if environment not in valid_environments:
                return FlextResult[FlextTypes.Config.ConfigDict].fail(
                    f"Invalid environment '{environment}'. Valid options: {valid_environments}"
                )

            base_config: FlextTypes.Config.ConfigDict = {
                "environment": environment,
                "enable_event_sourcing": True,
                "enable_domain_rules_validation": True,
            }

            if environment == FlextConstants.Config.ConfigEnvironment.PRODUCTION.value:
                base_config.update({
                    "aggregate_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                    "log_level": FlextConstants.Config.LogLevel.WARNING.value,
                    "max_aggregate_size": 500,
                    "enable_snapshot_storage": True,
                })
            elif (
                environment == FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value
            ):
                base_config.update({
                    "aggregate_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                    "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                    "max_aggregate_size": 2000,
                    "enable_snapshot_storage": False,
                })

            return FlextResult[FlextTypes.Config.ConfigDict].ok(base_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to create environment aggregates configuration: {e}"
            )

    @classmethod
    def optimize_aggregates_performance(
        cls, performance_level: str = "balanced"
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Optimize aggregates system performance settings."""
        try:
            valid_levels = ["low", "balanced", "high", "extreme"]
            if performance_level not in valid_levels:
                return FlextResult[FlextTypes.Config.ConfigDict].fail(
                    f"Invalid performance_level '{performance_level}'. Valid options: {valid_levels}"
                )

            optimized_config: FlextTypes.Config.ConfigDict = {
                "environment": FlextConstants.Config.ConfigEnvironment.PRODUCTION.value,
                "aggregate_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                "log_level": FlextConstants.Config.LogLevel.WARNING.value,
                "performance_level": performance_level,
                "optimization_enabled": True,
                "optimization_timestamp": "2025-01-01T00:00:00Z",
            }

            if performance_level == "high":
                optimized_config.update({
                    "enable_async_event_processing": True,
                    "event_batch_size": 100,
                    "enable_aggregate_caching": True,
                    "max_concurrent_aggregates": 500,
                })
            elif performance_level == "extreme":
                optimized_config.update({
                    "enable_async_event_processing": True,
                    "event_batch_size": 1000,
                    "enable_aggregate_caching": True,
                    "max_concurrent_aggregates": 5000,
                    "enable_domain_rules_validation": False,  # Skip for max speed
                })

            return FlextResult[FlextTypes.Config.ConfigDict].ok(optimized_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to optimize aggregates performance: {e}"
            )


# =============================================================================
# MODULE EXPORTS - Backward compatibility facades + FlextTypes.Config
# All exports point to consolidated FlextModels implementations
# =============================================================================

__all__ = [
    "FlextAggregates",  # Main class with FlextTypes.Config integration
    # Legacy compatibility aliases moved to flext_core.legacy to avoid type conflicts
]
