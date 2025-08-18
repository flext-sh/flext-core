from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from _typeshed import Incomplete
from pydantic import SecretStr, SerializationInfo
from pydantic_settings import BaseSettings

from flext_core.models import FlextModel
from flext_core.result import FlextResult

__all__ = [
    "CONFIG_VALIDATION_MESSAGES",
    "DEFAULT_ENVIRONMENT",
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_PAGE_SIZE",
    "DEFAULT_RETRIES",
    "DEFAULT_TIMEOUT",
    "FlextAbstractConfig",
    "FlextBaseConfigModel",
    "FlextBaseConfigModel",
    "FlextConfig",
    "FlextConfigDefaults",
    "FlextConfigFactory",
    "FlextConfigOps",
    "FlextConfigValidation",
    "FlextDatabaseConfig",
    "FlextJWTConfig",
    "FlextLDAPConfig",
    "FlextObservabilityConfig",
    "FlextOracleConfig",
    "FlextRedisConfig",
    "FlextSettings",
    "FlextSingerConfig",
    "_BaseConfigValidation",
    "create_config",
    "load_config_from_env",
    "load_config_from_file",
    "merge_configs",
    "safe_get_env_var",
    "safe_load_json_file",
    "validate_config",
]

class FlextSettings(BaseSettings):
    model_config: ClassVar[Incomplete]
    def validate_business_rules(self) -> FlextResult[None]: ...
    def serialize_settings_for_api(
        self,
        serializer: Callable[[FlextSettings], dict[str, object]],
        info: SerializationInfo,
    ) -> dict[str, object]: ...
    @classmethod
    def create_with_validation(
        cls, overrides: dict[str, object] | None = None, **kwargs: object
    ) -> FlextResult[FlextSettings]: ...

class FlextBaseConfigModel(FlextSettings): ...

class FlextConfig(FlextModel):
    name: str
    version: str
    description: str
    environment: str
    debug: bool
    log_level: str
    timeout: int
    retries: int
    page_size: int
    enable_caching: bool
    enable_metrics: bool
    enable_tracing: bool
    @classmethod
    def validate_environment(cls, v: str) -> str: ...
    @classmethod
    def validate_log_level(cls, v: str) -> str: ...
    @classmethod
    def validate_positive_integers(cls, v: int) -> int: ...
    def validate_business_rules(self) -> FlextResult[None]: ...
    def serialize_environment(self, value: str) -> dict[str, object]: ...
    def serialize_log_level(self, value: str) -> dict[str, object]: ...
    def serialize_config_for_api(
        self,
        serializer: Callable[[FlextConfig], dict[str, object]],
        info: SerializationInfo,
    ) -> dict[str, object]: ...
    @classmethod
    def create_complete_config(
        cls,
        config_data: dict[str, object],
        *,
        apply_defaults: bool = True,
        validate_all: bool = True,
    ) -> FlextResult[dict[str, object]]: ...
    @classmethod
    def load_and_validate_from_file(
        cls, file_path: str, required_keys: list[str] | None = None
    ) -> FlextResult[dict[str, object]]: ...
    @classmethod
    def safe_load_from_dict(
        cls, config_data: dict[str, object]
    ) -> FlextResult[dict[str, object]]: ...
    @classmethod
    def merge_and_validate_configs(
        cls, base_config: dict[str, object], override_config: dict[str, object]
    ) -> FlextResult[dict[str, object]]: ...
    @classmethod
    def get_env_with_validation(
        cls,
        env_var: str,
        *,
        validate_type: type = ...,
        default: object = None,
        required: bool = False,
    ) -> FlextResult[object]: ...
    @classmethod
    def merge_configs(
        cls, base_config: dict[str, object], override_config: dict[str, object]
    ) -> FlextResult[dict[str, object]]: ...
    @classmethod
    def validate_config_value(
        cls, value: object, validator: object, error_message: str = "Validation failed"
    ) -> FlextResult[bool]: ...
    @staticmethod
    def get_model_config(
        description: str = "Base configuration model",
        *,
        frozen: bool = True,
        extra: str = "forbid",
        validate_assignment: bool = True,
        use_enum_values: bool = True,
        str_strip_whitespace: bool = True,
        validate_all: bool = True,
        allow_reuse: bool = True,
    ) -> dict[str, object]: ...

class FlextDatabaseConfig(FlextModel):
    host: str
    port: int
    database: str
    username: str
    password: SecretStr
    database_schema: str
    pool_size: int
    max_overflow: int
    pool_timeout: int
    echo: bool
    @classmethod
    def validate_port(cls, v: int) -> int: ...
    @classmethod
    def validate_non_empty_string(cls, v: str) -> str: ...
    def validate_business_rules(self) -> FlextResult[None]: ...
    def get_connection_string(self) -> str: ...
    def serialize_password(self, value: SecretStr) -> dict[str, object]: ...
    def serialize_connection_fields(self, value: str) -> dict[str, object]: ...
    def serialize_database_config_for_api(
        self,
        serializer: Callable[[FlextDatabaseConfig], dict[str, object]],
        info: SerializationInfo,
    ) -> dict[str, object]: ...
    def to_database_dict(self) -> dict[str, object]: ...

class FlextRedisConfig(FlextModel):
    host: str
    port: int
    password: SecretStr
    database: int
    decode_responses: bool
    socket_timeout: int
    connection_pool_max_connections: int
    @classmethod
    def validate_database(cls, v: int) -> int: ...
    def get_connection_string(self) -> str: ...
    def serialize_redis_database(self, value: int) -> dict[str, object]: ...
    def serialize_redis_config_for_api(
        self,
        serializer: Callable[[FlextRedisConfig], dict[str, object]],
        info: SerializationInfo,
    ) -> dict[str, object]: ...
    def to_redis_dict(self) -> dict[str, object]: ...

class FlextLDAPConfig(FlextModel):
    server: str
    port: int
    use_ssl: bool
    use_tls: bool
    bind_dn: str
    bind_password: SecretStr
    base_dn: str
    search_base: str
    search_filter: str
    attributes: list[str]
    timeout: int
    @classmethod
    def validate_base_dn(cls, v: str) -> str: ...
    @classmethod
    def validate_ldap_port(cls, v: int) -> int: ...
    def validate_business_rules(self) -> FlextResult[None]: ...
    def get_connection_string(self) -> str: ...
    def serialize_bind_password(self, value: SecretStr) -> dict[str, object]: ...
    def serialize_security_flags(self, value: bool) -> dict[str, object]: ...
    def serialize_ldap_config_for_api(
        self,
        serializer: Callable[[FlextLDAPConfig], dict[str, object]],
        info: SerializationInfo,
    ) -> dict[str, object]: ...
    def to_ldap_dict(self) -> dict[str, object]: ...
    @property
    def host(self) -> str: ...

class FlextOracleConfig(FlextModel):
    host: str
    port: int
    service_name: str | None
    sid: str | None
    username: str
    password: SecretStr
    oracle_schema: str
    pool_min: int
    pool_max: int
    connection_timeout: int
    def model_post_init(self, __context: object, /) -> None: ...
    def validate_business_rules(self) -> FlextResult[None]: ...
    def get_connection_string(self) -> str: ...
    def serialize_oracle_password(self, value: SecretStr) -> dict[str, object]: ...
    def serialize_oracle_identifier(self, value: str | None) -> dict[str, object]: ...
    def serialize_oracle_config_for_api(
        self,
        serializer: Callable[[FlextOracleConfig], dict[str, object]],
        info: SerializationInfo,
    ) -> dict[str, object]: ...
    def to_oracle_dict(self) -> dict[str, object]: ...

class FlextJWTConfig(FlextModel):
    secret_key: SecretStr
    algorithm: str
    access_token_expire_minutes: int
    refresh_token_expire_days: int
    issuer: str
    audience: list[str]
    @classmethod
    def validate_algorithm(cls, v: str) -> str: ...
    @classmethod
    def validate_secret_key(cls, v: object) -> SecretStr: ...
    def to_jwt_dict(self) -> dict[str, object]: ...
    def serialize_jwt_secret(self, value: SecretStr) -> dict[str, object]: ...
    def serialize_jwt_algorithm(self, value: str) -> dict[str, object]: ...
    def serialize_jwt_config_for_api(
        self,
        serializer: Callable[[FlextJWTConfig], dict[str, object]],
        info: SerializationInfo,
    ) -> dict[str, object]: ...

class FlextObservabilityConfig(FlextModel):
    log_level: str
    log_format: str
    logging_enabled: bool
    logging_level: str
    logging_format: str
    tracing_enabled: bool
    service_name: str
    tracing_environment: str
    metrics_enabled: bool
    metrics_port: int
    metrics_path: str
    health_check_enabled: bool
    health_check_port: int
    health_check_path: str
    @classmethod
    def validate_log_level(cls, v: str) -> str: ...
    def to_observability_dict(self) -> dict[str, object]: ...
    ENABLE_METRICS: ClassVar[bool]
    TRACE_ENABLED: ClassVar[bool]
    TRACE_SAMPLE_RATE: ClassVar[float]

class FlextSingerConfig(FlextModel):
    tap_executable: str
    target_executable: str
    config_file: str
    catalog_file: str
    state_file: str
    output_file: str
    stream_name: str
    batch_size: int
    stream_schema: dict[str, object]
    stream_config: dict[str, object]
    @classmethod
    def validate_stream_name(cls, v: str) -> str: ...
    def to_singer_dict(self) -> dict[str, object]: ...

class FlextConfigFactory:
    @classmethod
    def create(cls, config_type: str, **kwargs: object) -> FlextResult[FlextModel]: ...
    @classmethod
    def create_from_env(
        cls, config_type: str, prefix: str = "FLEXT"
    ) -> FlextResult[FlextModel]: ...
    @classmethod
    def create_from_file(
        cls, config_type: str, file_path: str | Path
    ) -> FlextResult[FlextModel]: ...
    @classmethod
    def register_config(cls, name: str, config_class: type[FlextModel]) -> None: ...
    @classmethod
    def get_registered_types(cls) -> list[str]: ...

def create_config(config_type: str, **kwargs: object) -> FlextResult[FlextModel]: ...
def load_config_from_env(
    config_type: str, prefix: str = "FLEXT"
) -> FlextResult[FlextModel]: ...
def load_config_from_file(
    config_type: str, file_path: str | Path
) -> FlextResult[FlextModel]: ...
def safe_get_env_var(
    var_name: str, default: str = "", *, required: bool = False
) -> FlextResult[str]: ...
def safe_load_json_file(file_path: str | Path) -> FlextResult[dict[str, object]]: ...
def merge_configs(*configs: FlextModel) -> FlextResult[dict[str, object]]: ...
def validate_config(
    config: dict[str, object], schema: dict[str, object] | None = None
) -> FlextResult[None]: ...

class FlextConfigOps:
    @staticmethod
    def safe_load_from_dict(
        data: dict[str, object], required_keys: list[str] | None = None
    ) -> FlextResult[dict[str, object]]: ...
    @staticmethod
    def safe_get_env_var(
        var_name: str, default: str = "", *, required: bool = False
    ) -> FlextResult[str]: ...
    @staticmethod
    def safe_load_json_file(
        file_path: str | Path,
    ) -> FlextResult[dict[str, object]]: ...
    @staticmethod
    def safe_save_json_file(
        data: dict[str, object], file_path: str | Path, *, create_dirs: bool = False
    ) -> FlextResult[None]: ...

class FlextConfigDefaults:
    @staticmethod
    def apply_defaults(
        config: dict[str, object], defaults: dict[str, object]
    ) -> FlextResult[dict[str, object]]: ...

class FlextConfigValidation:
    @staticmethod
    def validate_config_value(
        value: object, validator: object, key: str = "field"
    ) -> FlextResult[bool]: ...
    @staticmethod
    def validate_config_type(
        value: object, expected_type: type, key_name: str = "field"
    ) -> FlextResult[bool]: ...
    @staticmethod
    def validate_config_range(
        value: float,
        min_val: float | None = None,
        max_val: float | None = None,
        key_name: str = "field",
    ) -> FlextResult[bool]: ...

_BaseConfigValidation = FlextConfigValidation
FlextAbstractConfig = FlextConfig
DEFAULT_TIMEOUT: int
DEFAULT_RETRIES: int
DEFAULT_PAGE_SIZE: int
DEFAULT_LOG_LEVEL: str
DEFAULT_ENVIRONMENT: str
CONFIG_VALIDATION_MESSAGES: Incomplete
