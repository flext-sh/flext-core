from typing import Protocol

from _typeshed import Incomplete

from flext_core.result import FlextResult

__all__ = [
    "FlextMixinDelegator",
    "create_mixin_delegator",
    "validate_delegation_system",
]

class _HasDelegator(Protocol):
    delegator: _DelegatorProtocol

class _DelegatorProtocol(Protocol):
    def get_delegation_info(self) -> dict[str, object]: ...

class FlextDelegatedProperty:
    prop_name: Incomplete
    mixin_instance: Incomplete
    has_setter: Incomplete
    __doc__: Incomplete
    def __init__(
        self,
        prop_name: str,
        mixin_instance: object,
        *,
        has_setter: bool,
        doc: str | None = None,
    ) -> None: ...
    def __get__(self, instance: object, owner: type | None = None) -> object: ...
    def __set__(self, instance: object, value: object) -> None: ...

class FlextMixinDelegator:
    def __init__(self, host_instance: object, *mixin_classes: type) -> None: ...
    def get_mixin_instance(self, mixin_class: type) -> object | None: ...
    def get_delegation_info(self) -> dict[str, object]: ...

def create_mixin_delegator(
    host_instance: object, *mixin_classes: type
) -> FlextMixinDelegator: ...
def validate_delegation_system() -> FlextResult[
    dict[str, str | list[str] | dict[str, object]]
]: ...
