from typing import NamedTuple

from _typeshed import Incomplete

from flext_core.typings import TAnyList

__all__ = [
    "AVAILABLE_FEATURES",
    "BUILD_TYPE",
    "MAX_PYTHON_VERSION",
    "MIN_PYTHON_VERSION",
    "RELEASE_DATE",
    "RELEASE_NAME",
    "VERSION_MAJOR",
    "VERSION_MINOR",
    "VERSION_PATCH",
    "FlextCompatibilityResult",
    "FlextVersionInfo",
    "__version__",
    "check_python_compatibility",
    "compare_versions",
    "get_available_features",
    "get_version_info",
    "get_version_string",
    "get_version_tuple",
    "is_feature_available",
    "validate_version_format",
]

__version__: Incomplete
VERSION_MAJOR: int
VERSION_MINOR: int
VERSION_PATCH: int
RELEASE_NAME: str
RELEASE_DATE: str
BUILD_TYPE: str
MIN_PYTHON_VERSION: Incomplete
MAX_PYTHON_VERSION: Incomplete
AVAILABLE_FEATURES: Incomplete

class FlextVersionInfo(NamedTuple):
    major: int
    minor: int
    patch: int
    release_name: str
    release_date: str
    build_type: str

class FlextCompatibilityResult(NamedTuple):
    is_compatible: bool
    current_version: tuple[int, ...]
    required_version: tuple[int, ...]
    error_message: str
    recommendations: TAnyList

def get_version_tuple() -> tuple[int, int, int]: ...
def get_version_info() -> FlextVersionInfo: ...
def get_version_string() -> str: ...
def check_python_compatibility() -> FlextCompatibilityResult: ...
def is_feature_available(feature_name: str) -> bool: ...
def get_available_features() -> TAnyList: ...
def compare_versions(version1: str, version2: str) -> int: ...
def validate_version_format(version: str) -> bool: ...
