from collections.abc import Callable as Callable
from typing import Literal

from pydantic import SecretStr

from flext_core import FlextConfig, FlextEntity, FlextResult, FlextValue

class FlextOracleConfig(FlextConfig):
    host: str
    port: int
    service_name: str | None
    sid: str | None
    username: str
    password: SecretStr
    max_connections: int
    def validate_business_rules(self) -> FlextResult[None]: ...

class FlextUserProfile(FlextValue):
    email: str
    full_name: str
    role: Literal["REDACTED_LDAP_BIND_PASSWORD", "user", "viewer"]
    preferences: dict[str, object]
    def validate_business_rules(self) -> FlextResult[None]: ...

class FlextDataPipeline(FlextEntity):
    name: str
    source_config: FlextOracleConfig
    owner: FlextUserProfile
    status: Literal["active", "inactive", "error"]
    processed_records: int
    def validate_business_rules(self) -> FlextResult[None]: ...
    def activate(self) -> FlextResult[None]: ...

def pipeline_factory() -> FlextDataPipeline: ...
def pipeline_validator(p: FlextDataPipeline) -> bool: ...

DatabaseConnection: str
UserCredentials: dict[str, str]
LoggerContext: dict[str, str]

class FlextPipelineService:
    def __init__(self) -> None: ...
    def create_pipeline(
        self,
        name: str,
        oracle_config: dict[str, object],
        owner_profile: dict[str, object],
    ) -> FlextResult[FlextDataPipeline]: ...
    def activate_pipeline(self, pipeline_id: str) -> FlextResult[str]: ...
    def get_pipeline_stats(self) -> dict[str, object]: ...

class FlextUnifiedUtilities:
    @staticmethod
    def validate_oracle_connection(
        connection_string: str,
    ) -> FlextResult[dict[str, str]]: ...
    @staticmethod
    def format_metric_display(metric: dict[str, object]) -> str: ...
    @staticmethod
    def safe_transform_data(
        data: dict[str, object],
        transformer: Callable[[dict[str, object]], dict[str, object]],
    ) -> FlextResult[dict[str, object]]: ...

async def demonstrate_foundation_models() -> FlextDataPipeline | None: ...
def demonstrate_semantic_types() -> None: ...
def demonstrate_domain_services(
    service: FlextPipelineService, pipeline: FlextDataPipeline
) -> None: ...
def demonstrate_utilities(service: FlextPipelineService) -> None: ...
def demonstrate_error_handling(service: FlextPipelineService) -> None: ...
def demonstrate_domain_events(pipeline: FlextDataPipeline) -> None: ...
def print_completion_summary() -> None: ...
async def demonstrate_unified_patterns() -> None: ...
def main() -> None: ...
