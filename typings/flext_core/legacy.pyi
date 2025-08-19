from _typeshed import Incomplete

from flext_core.config import FlextSettings
from flext_core.mixins import (
    FlextCacheableMixin,
    FlextCommandMixin,
    FlextComparableMixin,
    FlextIdentifiableMixin,
    FlextLoggableMixin,
    FlextSerializableMixin,
    FlextTimestampMixin,
    FlextTimingMixin,
    FlextValidatableMixin,
)
from flext_core.models import (
    FlextEntity,
    FlextFactory as FlextFactory,
    FlextModel,
    FlextValue,
)
from flext_core.observability import (
    FlextConsoleLogger,
    FlextInMemoryMetrics,
    FlextMinimalObservability,
    FlextNoOpTracer,
    FlextSimpleAlerts,
)
from flext_core.result import FlextResult
from flext_core.schema_processing import (
    FlextBaseConfigManager,
    FlextBaseEntry,
    FlextBaseFileWriter,
    FlextBaseProcessor,
    FlextBaseSorter,
    FlextConfigAttributeValidator,
    FlextEntryType,
)
from flext_core.utilities import FlextUtilities
from flext_core.validation import (
    FlextValidators,
)

__all__ = [
    "BaseConfigManager",
    "BaseEntry",
    "BaseFileWriter",
    "BaseProcessor",
    "BaseSorter",
    "ConfigAttributeValidator",
    "ConsoleLogger",
    "EntryType",
    "FlextBaseModel",
    "FlextBaseSettings",
    "FlextBaseUtilities",
    "FlextBaseValidators",
    "FlextConfiguration",
    "FlextDomainEntity",
    "FlextDomainValueObject",
    "FlextFactory",
    "FlextHelpers",
    "FlextImmutableModel",
    "FlextMutableModel",
    "FlextValidationUtils",
    "FlextValueObjectFactory",
    "InMemoryMetrics",
    "LegacyCompatibleCacheableMixin",
    "LegacyCompatibleCommandMixin",
    "LegacyCompatibleComparableMixin",
    "LegacyCompatibleDataMixin",
    "LegacyCompatibleEntityMixin",
    "LegacyCompatibleFullMixin",
    "LegacyCompatibleIdentifiableMixin",
    "LegacyCompatibleLoggableMixin",
    "LegacyCompatibleSerializableMixin",
    "LegacyCompatibleServiceMixin",
    "LegacyCompatibleTimestampMixin",
    "LegacyCompatibleTimingMixin",
    "LegacyCompatibleValidatableMixin",
    "MinimalObservability",
    "NoOpTracer",
    "SimpleAlerts",
    "_BaseConfigDefaults",
    "_BaseConfigValidation",
    "_PerformanceConfig",
    "check_python_compatibility",
    "compare_versions",
    "get_available_features",
    "get_version_info",
    "get_version_string",
    "get_version_tuple",
    "is_feature_available",
    "validate_version_format",
]

LegacyCompatibleCommandMixin = FlextCommandMixin
LegacyCompatibleComparableMixin = FlextComparableMixin
LegacyCompatibleDataMixin = FlextValidatableMixin
LegacyCompatibleEntityMixin = FlextIdentifiableMixin
LegacyCompatibleFullMixin = FlextValidatableMixin
LegacyCompatibleIdentifiableMixin = FlextIdentifiableMixin
LegacyCompatibleLoggableMixin = FlextLoggableMixin
LegacyCompatibleSerializableMixin = FlextSerializableMixin
LegacyCompatibleServiceMixin = FlextLoggableMixin
LegacyCompatibleTimestampMixin = FlextTimestampMixin
LegacyCompatibleTimingMixin = FlextTimingMixin
LegacyCompatibleValidatableMixin = FlextValidatableMixin
LegacyCompatibleValueObjectMixin = FlextValidatableMixin
InMemoryMetrics = FlextInMemoryMetrics
MinimalObservability = FlextMinimalObservability
NoOpTracer = FlextNoOpTracer
SimpleAlerts = FlextSimpleAlerts
BaseEntry = FlextBaseEntry
BaseFileWriter = FlextBaseFileWriter
BaseProcessor = FlextBaseProcessor
BaseSorter = FlextBaseSorter
ConfigAttributeValidator = FlextConfigAttributeValidator
EntryType = FlextEntryType

def check_python_compatibility() -> bool: ...
def compare_versions(v1: str, v2: str) -> int: ...
def get_available_features() -> list[str]: ...
def get_version_info() -> dict[str, str]: ...
def get_version_string() -> str: ...
def get_version_tuple() -> tuple[int, ...]: ...
def is_feature_available(feature: str) -> bool: ...
def validate_version_format(version_str: str) -> bool: ...

FlextDomainEntity = FlextEntity
FlextDomainValueObject = FlextValue
FlextBaseModel = FlextModel
FlextImmutableModel = FlextValue
FlextMutableModel = FlextEntity
FlextBaseSettings = FlextSettings
FlextConfiguration = FlextSettings
FlextBaseValidators = FlextValidators
FlextValidationUtils = FlextValidators
FlextBaseUtilities = FlextUtilities
FlextHelpers = FlextUtilities
ConsoleLogger = FlextConsoleLogger
BaseConfigManager = FlextBaseConfigManager
LegacyCompatibleCacheableMixin = FlextCacheableMixin
FlextValueObjectFactory = FlextFactory

class _BaseConfigDefaults:
    TIMEOUT: Incomplete
    RETRIES: Incomplete
    PAGE_SIZE: Incomplete

class _BaseConfigValidation:
    @staticmethod
    def validate_config(config: dict[str, object]) -> FlextResult[bool]: ...
    @staticmethod
    def validate_config_type(
        value: object, expected_type: type[object], key_name: str = "value"
    ) -> FlextResult[bool]: ...
    @staticmethod
    def validate_config_value(
        value: object, validator: object, error_message: str = "Validation failed"
    ) -> FlextResult[bool]: ...
    @staticmethod
    def validate_config_range(
        value: float,
        min_value: float | None = None,
        max_value: float | None = None,
        key_name: str = "value",
    ) -> FlextResult[bool]: ...

class _PerformanceConfig:
    TIMEOUT: Incomplete
    BATCH_SIZE: Incomplete
    MAX_CONNECTIONS: Incomplete
