from examples.shared_domain import (
    Order as SharedOrder,
    Product as SharedProduct,
    User as SharedUser,
)

from flext_core import (
    FlextCacheableMixin,
    FlextComparableMixin,
    FlextLoggableMixin,
    FlextSerializableMixin,
    FlextTimestampMixin,
)

from .formatting_helpers import (
    ADULT_AGE_THRESHOLD as ADULT_AGE_THRESHOLD,
    MIDDLE_AGED_THRESHOLD as MIDDLE_AGED_THRESHOLD,
    YOUNG_ADULT_AGE_THRESHOLD as YOUNG_ADULT_AGE_THRESHOLD,
)

class UtilityDemoUser(SharedUser, FlextCacheableMixin, FlextSerializableMixin):
    def get_cache_key(self) -> str: ...
    def get_cache_ttl(self) -> int: ...
    def serialize_key(self) -> str: ...
    def to_serializable(self) -> dict[str, object]: ...

class UtilityDemoProduct(
    SharedProduct,
    FlextCacheableMixin,
    FlextSerializableMixin,
    FlextComparableMixin,
    FlextTimestampMixin,
):
    def get_cache_key(self) -> str: ...
    def get_cache_ttl(self) -> int: ...
    def serialize_key(self) -> str: ...
    def to_serializable(self) -> dict[str, object]: ...
    def compare_with(self, other: object) -> int: ...

class UtilityDemoOrder(
    SharedOrder, FlextLoggableMixin, FlextSerializableMixin, FlextTimestampMixin
):
    def get_log_context(self) -> dict[str, object]: ...
    def serialize_key(self) -> str: ...
    def to_serializable(self) -> dict[str, object]: ...
