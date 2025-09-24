# FLEXT Pydantic Standardization Plan

## Current State Analysis

**✅ ALREADY IMPLEMENTED CORRECTLY:**

- flext-core: FlextConfig, FlextModels, FlextConstants with Pydantic 2.11 features
- flext-cli: FlextCliConfig extends FlextConfig, FlextCliModels extends FlextModels, FlextCliConstants extends FlextConstants
- flext-api: FlextApiConfig extends FlextConfig, FlextApiModels extends FlextModels, FlextApiConstants extends FlextConstants
- algar-oud-mig: AlgarOudMigConfig extends FlextConfig, AlgarOudMigModels extends FlextModels, AlgarOudMigConstants extends FlextConstants

## Standardization Requirements

### 1. Project Structure Pattern

Every FLEXT ecosystem project MUST have:

- `{ProjectName}Config` extending `FlextConfig`
- `{ProjectName}Models` extending `FlextModels`
- `{ProjectName}Constants` extending `FlextConstants`

### 2. Pydantic 2.11 Features to Implement

- `validate_return=True` for return type validation
- `arbitrary_types_allowed=True` for custom types
- `serialize_by_alias=True` for consistent serialization
- `populate_by_name=True` for flexible field population
- `ser_json_timedelta="iso8601"` for datetime serialization
- `ser_json_bytes="base64"` for binary data
- `nested_model_default_partial_update=True` for nested models
- `enable_decoding=True` for environment variables
- `cli_parse_args=False` and `cli_avoid_json=True` for settings

### 3. Model Configuration Standards

- All models MUST use `FlextModels.ArbitraryTypesModel` as base
- All models MUST have proper Field defaults from Constants
- All models MUST use `FlextResult[T]` for validation
- All models MUST have comprehensive field validators

### 4. Constants Standards

- All constants MUST extend `FlextConstants`
- All defaults MUST be defined in Constants classes
- All enums MUST use `StrEnum` for better serialization
- All validation lists MUST be centralized in Constants

### 5. Configuration Standards

- All configs MUST extend `FlextConfig`
- All configs MUST use environment variable prefixes
- All configs MUST have proper Field descriptions
- All configs MUST use Constants for defaults

## Implementation Priority

1. Enhance flext-core with additional Pydantic 2.11 features
2. Standardize all ecosystem projects
3. Validate inheritance patterns
4. Enforce model-based data transport
5. Create comprehensive documentation
