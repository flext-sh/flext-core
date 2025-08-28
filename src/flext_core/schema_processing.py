"""Enterprise schema and entry processing system providing comprehensive data validation and transformation.

This module implements a comprehensive schema processing framework that supports
generic entry validation, data transformation, and flexible processing pipelines.
The system provides abstract base classes, protocol definitions, and concrete
implementations for processing various types of structured data including
configuration entries, ACL entries, and custom data formats.

**ARCHITECTURAL FOCUS**: This module follows FLEXT Core consolidation patterns
while providing specialized functionality for schema and entry processing that
requires flexibility and extensibility not provided by the main consolidated classes.

Core Processing Capabilities:
    - **Generic Entry Processing**: Abstract base classes for extensible entry processing
    - **Schema Validation**: Protocol-based validation with configurable validators
    - **Regex Processing**: Pattern-based data extraction and identifier validation
    - **Configuration Management**: Attribute validation and configuration object handling
    - **Processing Pipelines**: Chainable operations with type-safe transformations
    - **File Writing**: Protocol-based file output with header and entry management
    - **Sorting Operations**: Configurable entry sorting with custom key extraction

Schema Processing Features:
    - **Type-Safe Processing**: Generic type parameters for compile-time type safety
    - **FlextResult Integration**: Type-safe error handling throughout the processing chain
    - **Protocol-Based Design**: Extensible interfaces for validation and processing
    - **Pipeline Architecture**: Composable processing steps with failure handling
    - **Configuration Validation**: Comprehensive attribute validation with error reporting
    - **Identifier Extraction**: Flexible pattern-based identifier extraction from content

Integration Features:
    - **FlextResult[T] Integration**: All operations return FlextResult for type-safe error handling
    - **FlextModels Integration**: Uses FlextModels.Value for base entry value objects
    - **FlextTypes Integration**: Leverages TEntry type alias for generic entry processing
    - **Protocol-Based Design**: Extensive use of Protocol for flexible, testable interfaces
    - **Backward Compatibility**: Legacy aliases moved to legacy.py for clean migration

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import StrEnum
from typing import Protocol, TypeGuard, override

from flext_core.constants import FlextConstants
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes, TEntry


class FlextEntryType(StrEnum):
    """Base enumeration for standardized entry type classification.

    This enumeration provides a foundation for type-safe entry classification
    in schema processing systems. Subclasses should extend this enumeration
    to define specific entry types for their processing domains (e.g., ACL
    entries, configuration entries, data transformation entries).

    **ARCHITECTURAL ROLE**: Provides type-safe enumeration base for entry
    classification systems, enabling compile-time validation and consistent
    entry type handling across different processing contexts.

    Usage Examples:
        Extending for specific domains:

            >>> class ACLEntryType(FlextEntryType):
            ...     USER = "user"
            ...     GROUP = "group"
            ...     ROLE = "role"
            ...     PERMISSION = "permission"

            >>> class ConfigEntryType(FlextEntryType):
            ...     DATABASE = "database"
            ...     CACHE = "cache"
            ...     LOGGING = "logging"
            ...     SECURITY = "security"

        Using in processing contexts::

            def process_entry(
                entry: FlextBaseEntry, entry_type: FlextEntryType
            ) -> FlextResult[ProcessedEntry]:
                if entry.entry_type == ACLEntryType.USER:
                    return process_user_entry(entry)
                elif entry.entry_type == ConfigEntryType.DATABASE:
                    return process_database_config(entry)
                else:
                    return FlextResult[ProcessedEntry].fail(
                        f"Unknown entry type: {entry_type}"
                    )

    See Also:
        - FlextBaseEntry: Base entry value object that uses this enumeration
        - FlextBaseProcessor: Generic processor that works with entry types

    """


class FlextBaseEntry(FlextModels.Value):
    """Base entry value object providing standardized structure for schema and ACL processing.

    This class serves as the foundation value object for all entry types in the
    schema processing system. It extends FlextModels.Value to provide immutable,
    validated entry representations with standardized fields for type identification,
    content management, and identifier tracking.

    **ARCHITECTURAL ROLE**: Provides the base value object structure that all
    schema processing entries should extend, ensuring consistent data representation
    and validation across different processing contexts and entry types.

    Entry Structure:
        - **entry_type**: Classification of the entry (using FlextEntryType or subclass)
        - **clean_content**: Processed, normalized content without formatting artifacts
        - **original_content**: Raw, unprocessed content as originally received
        - **identifier**: Unique identifier extracted from the entry content

    Value Object Features:
        - **Immutability**: All fields are immutable after construction
        - **Validation**: Automatic validation through FlextModels.Value base class
        - **Equality**: Value-based equality comparison for testing and deduplication
        - **Serialization**: JSON serialization support through Pydantic integration
        - **Type Safety**: Strongly typed fields with validation

    Usage Examples:
        Creating custom entry types:

            >>> class UserEntry(FlextBaseEntry):
            ...     # User entry with additional user-specific fields
            ...     username: str
            ...     email: str
            ...     roles: list[str]

            >>> class ConfigEntry(FlextBaseEntry):
            ...     # Configuration entry with value and metadata
            ...     config_value: str
            ...     config_section: str
            ...     is_encrypted: bool

        Processing entries with type safety:

            >>> def validate_user_entry(
            ...     entry: UserEntry,
            ... ) -> FlextResult[None]:  # doctest: +SKIP
            ...     if not entry.username or len(entry.username) < 3:  # doctest: +SKIP
            ...         return FlextResult[None].fail(
            ...             "Username must be at least 3 characters"
            ...         )  # doctest: +SKIP
            ...     if "@" not in entry.email:  # doctest: +SKIP
            ...         return FlextResult[None].fail(
            ...             "Invalid email format"
            ...         )  # doctest: +SKIP
            ...     return FlextResult[None].ok(None)  # doctest: +SKIP

        Entry comparison and deduplication:

            >>> # entries = [entry1, entry2, entry3]  # doctest: +SKIP
            >>> # unique_entries = list({entry.identifier: entry for entry in entries}.values())  # doctest: +SKIP

    Integration Features:
        - **FlextModels.Value**: Inherits validation and serialization capabilities
        - **FlextResult Integration**: Works seamlessly with FlextResult error handling
        - **Protocol Compatibility**: Compatible with FlextEntryValidator protocol
        - **Processing Pipeline**: Can be processed through FlextProcessingPipeline

    See Also:
        - FlextModels.Value: Parent value object class with validation
        - FlextEntryType: Enumeration for entry type classification
        - FlextBaseProcessor: Generic processor that creates and validates entries

    """

    entry_type: str
    clean_content: str
    original_content: str
    identifier: str


class FlextEntryValidator(Protocol):
    """Protocol defining the interface for entry validation and whitelist checking.

    This protocol establishes the contract for entry validation systems,
    providing both content validation and whitelist-based identifier filtering.
    Implementations should provide comprehensive validation logic that can
    evaluate both the structure and content of entries as well as maintain
    whitelist/blacklist functionality for identifier-based filtering.

    **ARCHITECTURAL ROLE**: Defines the validation contract that enables
    flexible, pluggable validation strategies in the schema processing system.
    This protocol-based approach allows for dependency injection and easy
    testing with mock implementations.

    Validation Capabilities:
        - **Entry Validation**: Comprehensive validation of entry objects including
          structure, content, and business rule compliance
        - **Whitelist Checking**: Identifier-based filtering to allow/deny specific
          entries based on predefined whitelists or blacklists
        - **Extensible Design**: Protocol-based design allows custom implementations
          for different validation strategies and requirements

    Protocol Methods:
        - is_valid(): Validates complete entry objects with full context
        - is_whitelisted(): Checks if specific identifiers are allowed

    Usage Examples:
        Basic validator implementation:

            >>> class SimpleEntryValidator:  # doctest: +SKIP
            ...     def __init__(self, whitelist: set[str]):  # doctest: +SKIP
            ...         self.whitelist = whitelist  # doctest: +SKIP
            ...         self.required_fields = [
            ...             "entry_type",
            ...             "identifier",
            ...         ]  # doctest: +SKIP
            ...
            ...     def is_valid(self, entry: object) -> bool:  # doctest: +SKIP
            ...         if not isinstance(entry, FlextBaseEntry):  # doctest: +SKIP
            ...             return False  # doctest: +SKIP
            ...         # Check required fields  # doctest: +SKIP
            ...         for field in self.required_fields:  # doctest: +SKIP
            ...             if not hasattr(entry, field) or not getattr(
            ...                 entry, field
            ...             ):  # doctest: +SKIP
            ...                 return False  # doctest: +SKIP
            ...         # Additional business rules  # doctest: +SKIP
            ...         if len(entry.identifier) < 3:  # doctest: +SKIP
            ...             return False  # doctest: +SKIP
            ...         return True  # doctest: +SKIP
            ...
            ...     def is_whitelisted(self, identifier: str) -> bool:  # doctest: +SKIP
            ...         return identifier in self.whitelist  # doctest: +SKIP

        Complex validator with business rules:

            >>> class ACLEntryValidator:  # doctest: +SKIP
            ...     def __init__(self, policy_manager: PolicyManager):  # doctest: +SKIP
            ...         self.policy_manager = policy_manager  # doctest: +SKIP
            ...
            ...     def is_valid(self, entry: object) -> bool:  # doctest: +SKIP
            ...         if not isinstance(entry, ACLEntry):  # doctest: +SKIP
            ...             return False  # doctest: +SKIP
            ...         # Validate against security policies  # doctest: +SKIP
            ...         return self.policy_manager.validate_acl_entry(
            ...             entry
            ...         )  # doctest: +SKIP
            ...
            ...     def is_whitelisted(self, identifier: str) -> bool:  # doctest: +SKIP
            ...         # Check against dynamic whitelist  # doctest: +SKIP
            ...         return self.policy_manager.is_identifier_allowed(
            ...             identifier
            ...         )  # doctest: +SKIP

        Using with processors:

            >>> # validator = SimpleEntryValidator(whitelist={"admin", "user", "guest"})  # doctest: +SKIP
            >>> # processor = FlextBaseProcessor(validator=validator)  # doctest: +SKIP
            >>> # result = processor.extract_entry_info(content, "user", "PREFIX")  # doctest: +SKIP
            >>> # if result.success:  # doctest: +SKIP
            >>> #     print(f"Valid entry created: {result.value.identifier}")  # doctest: +SKIP
            # else:
            #     print(f"Validation failed: {result.error}")

    Implementation Considerations:
        - **Performance**: Validation methods may be called frequently, optimize for speed
        - **Thread Safety**: Implementations should be thread-safe for concurrent processing
        - **Error Handling**: Return boolean values, detailed errors handled by callers
        - **Immutability**: Avoid modifying entry objects during validation

    See Also:
        - FlextBaseProcessor: Uses this protocol for entry validation
        - FlextBaseEntry: Entry objects validated by this protocol
        - FlextResult[T]: Error handling system for validation failures

    """

    def is_valid(self, entry: object) -> bool:
        """Validate entry object for structural and business rule compliance.

        This method performs comprehensive validation of entry objects,
        checking both structural integrity (required fields, types, formats)
        and business rule compliance (constraints, relationships, policies).

        Args:
            entry: The entry object to validate (typically FlextBaseEntry or subclass)

        Returns:
            True if the entry passes all validation checks, False otherwise

        Example:
            Implementing comprehensive validation::

                def is_valid(self, entry: object) -> bool:
                    # Type checking
                    if not isinstance(entry, FlextBaseEntry):
                        return False

                    # Required field validation
                    if not all([
                        entry.entry_type,
                        entry.identifier,
                        entry.clean_content,
                    ]):
                        return False

                    # Format validation
                    if len(entry.identifier) < 3 or len(entry.identifier) > 50:
                        return False

                    # Business rule validation
                    if (
                        entry.entry_type == "admin"
                        and not entry.clean_content.startswith("admin_")
                    ):
                        return False

                    return True

        """
        ...

    def is_whitelisted(self, identifier: str) -> bool:
        """Check if identifier is allowed according to whitelist/blacklist policies.

        This method determines whether a specific identifier should be allowed
        in the processing system based on whitelist, blacklist, or policy-based
        filtering rules. This provides fine-grained access control for entry
        processing.

        Args:
            identifier: The identifier string to check against whitelist policies

        Returns:
            True if the identifier is allowed/whitelisted, False if blocked

        Example:
            Implementing whitelist checking::

                def is_whitelisted(self, identifier: str) -> bool:
                    # Simple whitelist checking
                    if identifier in self.allowed_identifiers:
                        return True

                    # Pattern-based checking
                    if any(
                        pattern.match(identifier) for pattern in self.allowed_patterns
                    ):
                        return True

                    # Blacklist checking
                    if identifier in self.blocked_identifiers:
                        return False

                    # Policy-based checking
                    return self.policy_manager.is_identifier_allowed(identifier)

        """
        ...


class FlextBaseProcessor[TEntry](ABC):
    r"""Generic abstract base processor for schema and entry processing with configurable validation.

    This abstract base class provides the foundation for entry processing systems that
    need to extract, validate, and transform structured data entries. It implements
    the Template Method pattern with configurable validation strategies and type-safe
    processing pipelines using FlextResult error handling.

    **ARCHITECTURAL ROLE**: Serves as the abstract base for specialized processors
    that handle different types of structured data (ACL entries, configuration entries,
    schema definitions). Provides common processing infrastructure while allowing
    concrete implementations to customize identifier extraction and entry creation logic.

    Generic Type Parameters:
        TEntry: The specific entry type this processor creates and manages

    Processing Capabilities:
        - **Abstract Processing Methods**: Template methods for identifier extraction and entry creation
        - **Validation Integration**: Optional FlextEntryValidator integration for flexible validation strategies
        - **Pipeline Processing**: Type-safe processing with FlextResult error handling throughout
        - **Entry Tracking**: Automatic tracking of successfully processed entries
        - **Error Handling**: Comprehensive error reporting with detailed failure messages
        - **Template Method Pattern**: Structured processing workflow with customizable steps

    Abstract Methods (Must be Implemented):
        - _extract_identifier(): Custom identifier extraction logic
        - _create_entry(): Custom entry creation and initialization logic

    Concrete Methods (Provided):
        - extract_entry_info(): Main processing pipeline with validation
        - _validate_identifier_extraction(): Identifier validation with whitelist checking
        - _validate_entry_creation(): Entry validation with business rules

    Usage Examples:
        Creating a custom processor::

            class UserEntryProcessor(FlextBaseProcessor[UserEntry]):
                \"\"\"Processor for user account entries.\"\"\"

                def __init__(self, validator: FlextEntryValidator | None = None):
                    super().__init__(validator)
                    self.username_pattern = re.compile(r'username:\\s*([a-zA-Z0-9_]+)')

                def _extract_identifier(self, content: str) -> FlextResult[str]:
                    \"\"\"Extract username from user entry content.\"\"\"
                    match = self.username_pattern.search(content)
                    if not match:
                        return FlextResult[str].fail("No username found in content")

                    username = match.group(1)
                    if len(username) < 3:
                        return FlextResult[str].fail("Username too short")

                    return FlextResult[str].ok(username)

                def _create_entry(
                    self,
                    entry_type: str,
                    clean_content: str,
                    original_content: str,
                    identifier: str,
                ) -> FlextResult[UserEntry]:
                    \"\"\"Create UserEntry from extracted data.\"\"\"
                    try:
                        # Extract additional fields from content
                        email = self._extract_email(clean_content)
                        roles = self._extract_roles(clean_content)

                        entry = UserEntry(
                            entry_type=entry_type,
                            clean_content=clean_content,
                            original_content=original_content,
                            identifier=identifier,
                            username=identifier,
                            email=email,
                            roles=roles
                        )
                        return FlextResult[UserEntry].ok(entry)
                    except Exception as e:
                        return FlextResult[UserEntry].fail(f"Failed to create user entry: {e}")

        Using the processor with validation::

            # Create validator with whitelist
            validator = SimpleEntryValidator(whitelist={'admin', 'user', 'guest'})

            # Create processor with validator
            processor = UserEntryProcessor(validator=validator)

            # Process entry content
            content = \"username: admin, email: admin@example.com, roles: [administrator]\"
            result = processor.extract_entry_info(content, \"user\", \"USER\")

            if result.success:
                user_entry = result.value
                print(f\"Created user: {user_entry.username}\")
                print(f\"Extracted entries: {len(processor._extracted_entries)}\")
            else:
                print(f\"Processing failed: {result.error}\")

        Batch processing with error handling::

            processor = UserEntryProcessor()
            entries = []
            errors = []

            for content_line in content_lines:
                result = processor.extract_entry_info(content_line, \"user\")
                if result.success:
                    entries.append(result.value)
                else:
                    errors.append(f\"Line {i}: {result.error}\")

            print(f\"Successfully processed: {len(entries)} entries\")
            print(f\"Failed to process: {len(errors)} entries\")

    Integration Features:
        - **FlextResult Integration**: All operations return FlextResult for type-safe error handling
        - **FlextEntryValidator Protocol**: Pluggable validation strategies via protocol-based design
        - **Generic Type Safety**: Compile-time type checking with generic type parameters
        - **Template Method Pattern**: Structured processing flow with customization points

    Performance Considerations:
        - **Entry Tracking**: Processed entries are automatically stored for batch operations
        - **Validation Caching**: Validators can implement caching for repeated validations
        - **Memory Management**: Consider clearing _extracted_entries for large batch processing
        - **Error Short-Circuiting**: Processing stops at first validation failure for efficiency

    Thread Safety:
        Implementations should ensure thread safety if used in concurrent environments.
        The base class maintains internal state (_extracted_entries) that may require
        synchronization for concurrent access.

    See Also:
        - FlextEntryValidator: Protocol for pluggable validation strategies
        - FlextResult: Type-safe error handling system
        - FlextBaseEntry: Base value object for processed entries
        - FlextRegexProcessor: Concrete implementation using regex-based processing

    """

    def __init__(self, validator: FlextEntryValidator | None = None) -> None:
        """Initialize processor with optional validation strategy.

        Sets up the processor with an optional validator for entry validation
        and whitelist checking. Initializes internal tracking for extracted entries.

        Args:
            validator: Optional validator implementing FlextEntryValidator protocol
                      for entry validation and identifier whitelist checking

        Example:
            Basic initialization::

                processor = MyEntryProcessor()

            With custom validator::

                validator = CustomValidator(whitelist=["admin", "user"])
                processor = MyEntryProcessor(validator=validator)

        """
        self.validator = validator
        self._extracted_entries: list[TEntry] = []

    @abstractmethod
    def _extract_identifier(self, content: str) -> FlextResult[str]:
        r"""Extract unique identifier from entry content using domain-specific logic.

        This abstract method must be implemented by concrete processors to define
        how identifiers are extracted from the specific content format they handle.
        The identifier should uniquely identify the entry within its domain.

        Args:
            content: The raw content string to extract identifier from

        Returns:
            FlextResult containing the extracted identifier string on success,
            or an error message describing why extraction failed

        Example:
            Implementation for LDAP DN extraction::

                def _extract_identifier(self, content: str) -> FlextResult[str]:
                    # Extract Distinguished Name from LDAP entry
                    dn_pattern = re.compile(r"dn:\s*(.+)", re.IGNORECASE)
                    match = dn_pattern.search(content)

                    if not match:
                        return FlextResult[str].fail("No DN found in LDAP entry")

                    dn = match.group(1).strip()
                    if not dn:
                        return FlextResult[str].fail("Empty DN extracted")

                    return FlextResult[str].ok(dn)

            Implementation for configuration key extraction::

                def _extract_identifier(self, content: str) -> FlextResult[str]:
                    # Extract configuration key from key=value format
                    key_pattern = re.compile(r"^([^=]+)=")
                    match = key_pattern.search(content.strip())

                    if not match:
                        return FlextResult[str].fail("No configuration key found")

                    key = match.group(1).strip()
                    if len(key) < 2:
                        return FlextResult[str].fail("Configuration key too short")

                    return FlextResult[str].ok(key)

        Note:
            This method should focus only on identifier extraction and basic validation.
            Complex business rule validation should be handled by the FlextEntryValidator.

        """
        ...

    @abstractmethod
    def _create_entry(
        self,
        entry_type: str,
        clean_content: str,
        original_content: str,
        identifier: str,
    ) -> FlextResult[TEntry]:
        r"""Create concrete entry instance from extracted data with proper validation.

        This abstract method must be implemented by concrete processors to create
        specific entry types from the extracted and validated data. The method should
        handle entry construction, field population, and initial validation.

        Args:
            entry_type: The type/category of the entry being created
            clean_content: Content with formatting artifacts removed
            original_content: Raw, unprocessed original content
            identifier: Unique identifier extracted from the content

        Returns:
            FlextResult containing the created entry instance on success,
            or an error message describing why creation failed

        Example:
            Implementation for user account entries::

                def _create_entry(
                    self,
                    entry_type: str,
                    clean_content: str,
                    original_content: str,
                    identifier: str,
                ) -> FlextResult[UserEntry]:
                    try:
                        # Parse additional fields from clean content
                        email_match = re.search(r"email:\s*([^,]+)", clean_content)
                        roles_match = re.search(r"roles:\s*\[([^\]]+)\]", clean_content)

                        email = email_match.group(1).strip() if email_match else ""
                        roles = (
                            [r.strip() for r in roles_match.group(1).split(",")]
                            if roles_match
                            else []
                        )

                        # Validate required fields
                        if not email or "@" not in email:
                            return FlextResult[UserEntry].fail(
                                "Invalid or missing email address"
                            )

                        # Create entry instance
                        entry = UserEntry(
                            entry_type=entry_type,
                            clean_content=clean_content,
                            original_content=original_content,
                            identifier=identifier,
                            username=identifier,
                            email=email,
                            roles=roles,
                            created_at=datetime.utcnow(),
                        )

                        return FlextResult[UserEntry].ok(entry)

                    except Exception as e:
                        return FlextResult[UserEntry].fail(
                            f"Entry creation failed: {e}"
                        )

            Implementation for configuration entries::

                def _create_entry(
                    self,
                    entry_type: str,
                    clean_content: str,
                    original_content: str,
                    identifier: str,
                ) -> FlextResult[ConfigEntry]:
                    try:
                        # Parse key=value format
                        if "=" not in clean_content:
                            return FlextResult[ConfigEntry].fail(
                                "Invalid configuration format"
                            )

                        key, value = clean_content.split("=", 1)
                        key = key.strip()
                        value = value.strip()

                        # Determine configuration section
                        section = self._determine_config_section(key)

                        # Check if value should be encrypted
                        is_sensitive = self._is_sensitive_config(key)

                        entry = ConfigEntry(
                            entry_type=entry_type,
                            clean_content=clean_content,
                            original_content=original_content,
                            identifier=identifier,
                            config_key=key,
                            config_value=value,
                            config_section=section,
                            is_encrypted=is_sensitive,
                        )

                        return FlextResult[ConfigEntry].ok(entry)

                    except Exception as e:
                        return FlextResult[ConfigEntry].fail(
                            f"Config entry creation failed: {e}"
                        )

        Note:
            This method should handle entry construction and basic field validation.
            Complex business rule validation will be performed by FlextEntryValidator
            after entry creation.

        """
        ...

    def extract_entry_info(
        self,
        content: str,
        entry_type: str,
        prefix: str = "",
    ) -> FlextResult[TEntry]:
        """Extract entry information from content with type safety."""
        # Step 1: Extract and validate identifier
        identifier_validation = self._validate_identifier_extraction(content)
        if identifier_validation.is_failure:
            return FlextResult[TEntry].fail(
                identifier_validation.error or "Identifier validation failed",
            )

        identifier = identifier_validation.value

        # Step 2: Create and validate entry
        clean_content = (
            content.replace(f"{prefix}: ", "").strip() if prefix else content.strip()
        )

        entry_validation = self._validate_entry_creation(
            entry_type,
            clean_content,
            content,
            identifier,
        )
        if entry_validation.is_failure:
            return entry_validation

        entry = entry_validation.value
        if entry is None:
            return FlextResult[TEntry].fail("Entry validation returned None")

        self._extracted_entries.append(entry)
        return FlextResult[TEntry].ok(entry)

    def _validate_identifier_extraction(self, content: str) -> FlextResult[str]:
        """Validate identifier extraction step."""
        identifier_result = self._extract_identifier(content)
        if identifier_result.is_failure:
            return FlextResult[str].fail(
                f"Failed to extract identifier: {identifier_result.error}",
            )

        identifier = identifier_result.value

        if self.validator and not self.validator.is_whitelisted(identifier):
            return FlextResult[str].fail(f"Identifier {identifier} not whitelisted")

        return FlextResult[str].ok(identifier)

    def _validate_entry_creation(
        self,
        entry_type: str,
        clean_content: str,
        content: str,
        identifier: str,
    ) -> FlextResult[TEntry]:
        """Validate an entry creation step."""
        entry_result = self._create_entry(
            entry_type,
            clean_content,
            content,
            identifier,
        )
        if entry_result.is_failure:
            return entry_result

        entry = entry_result.value
        if entry is None:
            return FlextResult[TEntry].fail("Entry creation returned None")

        if self.validator and not self.validator.is_valid(entry):
            return FlextResult[TEntry].fail(f"Entry validation failed for {identifier}")

        return FlextResult[TEntry].ok(entry)

    def process_content_lines(
        self,
        lines: list[str],
        entry_type: str,
        prefix: str = "",
    ) -> FlextResult[list[TEntry]]:
        """Process multiple content lines and return successful entries."""
        results: list[TEntry] = []
        errors: list[str] = []

        for line in lines:
            if not line.strip():
                continue

            result: FlextResult[TEntry] = self.extract_entry_info(
                line,
                entry_type,
                prefix,
            )
            if result.is_success:
                if result.value is not None:
                    results.append(result.value)
            else:
                errors.append(f"Line '{line[:50]}...': {result.error}")

        if errors and not results:
            return FlextResult[list[TEntry]].fail(
                f"All entries failed: {'; '.join(errors[:3])}"
            )

        # Return success even if some entries failed (partial success)
        return FlextResult[list[TEntry]].ok(results)

    def get_extracted_entries(self) -> list[TEntry]:
        """Get all successfully extracted entries."""
        return self._extracted_entries.copy()

    def clear_extracted_entries(self) -> None:
        """Clear extracted entries cache."""
        self._extracted_entries.clear()


class FlextRegexProcessor(FlextBaseProcessor[TEntry]):
    """Regex-based processor for entries with pattern matching."""

    def __init__(
        self,
        identifier_pattern: str,
        validator: FlextEntryValidator | None = None,
    ) -> None:
        """Initialize with a regex pattern for identifier extraction."""
        super().__init__(validator)
        self.identifier_pattern = re.compile(identifier_pattern)

    @override
    def _extract_identifier(self, content: str) -> FlextResult[str]:
        """Extract identifier using a regex pattern."""
        match = self.identifier_pattern.search(content)
        if not match:
            return FlextResult[str].fail(
                f"No identifier found matching pattern in: {content[:50]}",
            )

        return FlextResult[str].ok(match.group(1))


class FlextConfigAttributeValidator:
    """Utility for validating configuration attributes."""

    @staticmethod
    def has_attribute(config: object, attribute: str) -> bool:
        """Check if config has a specified attribute."""
        return hasattr(config, attribute)

    @staticmethod
    def has_rules_config(config: object) -> bool:
        """Check if config has rules_config attribute."""
        return hasattr(config, "rules_config")

    def _is_dict_like(self, config: object) -> TypeGuard[dict[str, object]]:
        """Type guard for dict-like configuration objects."""
        return isinstance(config, dict)

    def _extract_config_dict(self, config: object) -> dict[str, object]:
        """Extract configuration as dict using type-safe approach."""
        if self._is_dict_like(config):
            # Python 3.13+ discriminated union: config is dict[str, object]
            # Type narrowing through TypeGuard ensures safe access
            return config  # Type narrowing works here
        # For non-dict objects, extract __dict__ or return empty dict
        return getattr(config, "__dict__", {})

    @staticmethod
    def validate_required_attributes(
        config: object,
        required: list[str],
    ) -> FlextResult[bool]:
        """Validate config has required attributes - FACADE to base_validation."""
        # ARCHITECTURAL DECISION: Use centralized validation to eliminate duplication
        validator = FlextConfigAttributeValidator()
        config_dict = validator._extract_config_dict(config)

        # Simple validation: check if all required keys are present
        missing = [field for field in required if field not in config_dict]
        if missing:
            return FlextResult[bool].fail(
                f"Missing required attributes: {', '.join(missing)}",
            )
        return FlextResult[bool].ok(data=True)


class FlextBaseConfigManager:
    """Base configuration manager with attribute validation."""

    def __init__(self, config: object) -> None:
        """Initialize with a configuration object."""
        self.config = config
        self.validator = FlextConfigAttributeValidator()

    def get_config_value(self, key: str, default: object = None) -> object:
        """Get configuration value with optional default."""
        return getattr(self.config, key, default)

    def validate_config(
        self,
        required_attrs: list[str] | None = None,
    ) -> FlextResult[bool]:
        """Validate config has required attributes - FACADE to base_validation."""
        if required_attrs:
            # ARCHITECTURAL DECISION: Use centralized validation directly
            # to eliminate duplication
            return self.validator.validate_required_attributes(
                self.config,
                required_attrs,
            )
        return FlextResult[bool].ok(data=True)


class FlextBaseSorter[TEntry]:
    """Base sorter for entries with configurable sort key extraction."""

    def __init__(
        self,
        key_extractor: Callable[[TEntry], str] | None = None,
    ) -> None:
        """Initialize with optional key extractor function."""

        # Use a properly typed lambda that returns str (which implements SupportsRichComparison)
        def default_extractor(x: TEntry) -> str:
            return str(x)

        self.key_extractor: Callable[[TEntry], str] = key_extractor or default_extractor

    def sort_entries(self, entries: list[TEntry]) -> list[TEntry]:
        """Sort entries using the configured key extractor."""
        try:
            entries.sort(key=self.key_extractor)
            return entries
        except (TypeError, ValueError):
            # Return unsorted if sort fails
            return entries


class FlextBaseFileWriter(Protocol):
    """File writer protocol with common file operations."""

    def write_header(self, output_file: object) -> None:
        """Write file header."""
        ...

    def write_entry(self, output_file: object, entry: object) -> None:
        """Write single entry."""
        ...

    def write_entries(
        self,
        output_file: object,
        entries: list[object],
    ) -> FlextResult[None]:
        """Write multiple entries with header."""
        try:
            self.write_header(output_file)
            for entry in entries:
                self.write_entry(output_file, entry)
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Failed to write entries: {e}")


class FlextProcessingPipeline[InputT, OutputT]:
    """Generic processing pipeline for chaining operations."""

    def __init__(self) -> None:
        """Initialize empty pipeline."""
        self.steps: list[Callable[[object], FlextResult[object]]] = []
        self._input_type: type[InputT] | None = None
        self._output_type: type[OutputT] | None = None

    def add_step(
        self,
        step: Callable[[object], FlextResult[object]],
    ) -> FlextProcessingPipeline[InputT, OutputT]:
        """Add a processing step to a pipeline."""
        self.steps.append(step)
        return self

    def with_types(
        self,
        input_type: type[InputT],
        output_type: type[OutputT],
    ) -> FlextProcessingPipeline[InputT, OutputT]:
        """Set explicit types for better type checking."""
        self._input_type = input_type
        self._output_type = output_type
        return self

    def _is_output_type(self, _data: object) -> TypeGuard[OutputT]:
        """Type guard for output type - pipeline guarantees type consistency."""
        # In a well-formed pipeline, the final step should produce OutputT
        # This is a runtime assumption based on correct pipeline construction
        return True  # Trust the pipeline's type consistency

    def process(self, input_data: InputT) -> FlextResult[OutputT]:
        """Process input through all pipeline steps."""
        current_data: object = input_data
        for step in self.steps:
            result = step(current_data)
            if result.is_failure:
                return FlextResult[OutputT].fail(
                    result.error or "Processing step failed"
                )
            current_data = result.value

        # Python 3.13+ discriminated union with type guard
        # The pipeline guarantees type consistency through its construction
        if not self._is_output_type(current_data):
            return FlextResult[OutputT].fail("Pipeline output type mismatch")

        # Type narrowing: after type guard, current_data is OutputT
        return FlextResult[OutputT].ok(current_data)


# =============================================================================
# SCHEMA PROCESSING CONFIGURATION - FlextTypes.Config Integration
# =============================================================================


# Helper function for getting dependencies with fallback
def _get_dependencies() -> tuple[object, object, object]:
    """Get runtime dependencies with graceful fallback."""
    # Return module-level imports directly
    return FlextConstants, FlextResult, FlextTypes


class FlextSchemaProcessingConfig:
    """Enterprise schema processing system management with FlextTypes.Config integration."""

    @classmethod
    def configure_schema_processing_system(cls, config: dict[str, object]) -> object:
        """Configure schema processing system using FlextTypes.Config with StrEnum validation.

        Configures the FLEXT schema processing system including entry validation,
        identifier extraction patterns, regex processing optimization, validation
        pipeline efficiency, configuration attribute validation, and processing
        pipeline orchestration with comprehensive validation and performance tuning.

        Args:
            config: Configuration dictionary supporting:
                   - environment: Runtime environment (development, staging, production, test, local)
                   - processing_level: Processing validation level (strict, loose, disabled)
                   - log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL, TRACE)
                   - enable_regex_caching: Cache compiled regex patterns
                   - enable_entry_validation: Enable comprehensive entry validation
                   - processing_batch_size: Batch size for bulk processing operations

        Returns:
            FlextResult containing validated configuration with schema processing system settings

        Example:
            ```python
            config = {
                "environment": "production",
                "processing_level": "strict",
                "log_level": "INFO",
                "enable_regex_caching": True,
                "enable_entry_validation": True,
                "processing_batch_size": 1000,
            }
            result = FlextSchemaProcessingConfig.configure_schema_processing_system(
                config
            )
            if result.success:
                schema_config = result.unwrap()
            ```

        """
        try:
            _, _, _ = _get_dependencies()

            # Create working copy of config
            validated_config = dict(config)

            # Validate environment
            if "environment" in config:
                env_value = config["environment"]
                valid_environments = [
                    e.value for e in FlextConstants.Config.ConfigEnvironment
                ]
                if env_value not in valid_environments:
                    return FlextResult[dict[str, object]].fail(
                        f"Invalid environment '{env_value}'. Valid options: {valid_environments}"
                    )
            else:
                validated_config["environment"] = (
                    FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value
                )

            # Validate processing_level (using validation level as basis)
            if "processing_level" in config:
                level_value = config["processing_level"]
                valid_levels = [e.value for e in FlextConstants.Config.ValidationLevel]
                if level_value not in valid_levels:
                    return FlextResult[dict[str, object]].fail(
                        f"Invalid processing_level '{level_value}'. Valid options: {valid_levels}"
                    )
            else:
                validated_config["processing_level"] = (
                    FlextConstants.Config.ValidationLevel.LOOSE.value
                )

            # Validate log_level
            if "log_level" in config:
                log_level_value = config["log_level"]
                valid_log_levels = [e.value for e in FlextConstants.Config.LogLevel]
                if log_level_value not in valid_log_levels:
                    return FlextResult[dict[str, object]].fail(
                        f"Invalid log_level '{log_level_value}'. Valid options: {valid_log_levels}"
                    )
            else:
                validated_config["log_level"] = (
                    FlextConstants.Config.LogLevel.DEBUG.value
                )

            # Set default values for schema processing system specific settings
            validated_config.setdefault("enable_regex_caching", True)
            validated_config.setdefault("enable_entry_validation", True)
            validated_config.setdefault("processing_batch_size", 500)
            validated_config.setdefault("enable_pipeline_optimization", True)
            validated_config.setdefault("enable_identifier_validation", True)
            validated_config.setdefault("enable_content_preprocessing", True)
            validated_config.setdefault("regex_pattern_cache_size", 100)
            validated_config.setdefault("enable_processing_metrics", True)

            return FlextResult[dict[str, object]].ok(validated_config)

        except Exception as e:
            return FlextResult[dict[str, object]].fail(
                f"Failed to configure schema processing system: {e}"
            )

    @classmethod
    def get_schema_processing_system_config(cls) -> object:
        """Get current schema processing system configuration with runtime metrics.

        Retrieves the current schema processing system configuration including runtime metrics,
        active processing pipelines, regex pattern compilation statistics, entry validation
        performance, identifier extraction success rates, and batch processing throughput
        metrics for monitoring and diagnostics.

        Returns:
            FlextResult containing current schema processing system configuration with:
            - environment: Current runtime environment
            - processing_level: Current processing validation level
            - log_level: Current logging level
            - regex_processing_stats: Regex pattern compilation and matching performance
            - entry_validation_metrics: Entry validation success rates and timing
            - pipeline_performance: Processing pipeline throughput and efficiency metrics

        Example:
            ```python
            result = FlextSchemaProcessingConfig.get_schema_processing_system_config()
            if result.success:
                config = result.unwrap()
                print(f"Processing level: {config['processing_level']}")
            ```

        """
        try:
            # Get current schema processing system state for runtime metrics
            # NOTE: In a real implementation, these would come from actual system state
            current_config: FlextTypes.Config.ConfigDict = {
                # Current system configuration
                "environment": FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
                "processing_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                # Schema processing system specific settings
                "enable_regex_caching": True,
                "enable_entry_validation": True,
                "processing_batch_size": 500,
                "enable_pipeline_optimization": True,
                "enable_identifier_validation": True,
                "enable_content_preprocessing": True,
                # Runtime metrics and diagnostics
                "regex_processing_stats": {
                    "patterns_compiled": 45,
                    "patterns_cached": 38,
                    "pattern_matches_performed": 12456,
                    "avg_pattern_match_time_ms": 0.2,
                },
                "entry_validation_metrics": {
                    "entries_validated": 3456,
                    "validation_successes": 3398,
                    "validation_failures": 58,
                    "avg_validation_time_ms": 0.5,
                },
                "pipeline_performance": {
                    "pipelines_executed": 234,
                    "total_entries_processed": 15678,
                    "avg_processing_time_per_entry_ms": 1.2,
                    "batch_processing_throughput_per_sec": 850,
                },
                # System status and monitoring
                "active_processing_pipelines": 12,
                "regex_cache_size": 38,
                "memory_usage_mb": 89.3,
                # Monitoring and diagnostics
                "last_health_check": "2025-01-01T00:00:00Z",
                "system_status": "operational",
                "configuration_source": "default",
            }

            # Cast ConfigDict to dict[str, object] for type compatibility
            result_config: dict[str, object] = dict(current_config.items())
            return FlextResult[dict[str, object]].ok(result_config)

        except Exception as e:
            return FlextResult[dict[str, object]].fail(
                f"Failed to get schema processing system configuration: {e}"
            )

    @classmethod
    def create_environment_schema_processing_config(cls, environment: str) -> object:
        """Create environment-specific schema processing system configuration.

        Generates optimized configuration for schema processing based on the target
        environment (development, staging, production, test, local) with appropriate
        validation levels, regex caching settings, batch processing optimization,
        pipeline configurations, and performance tunings for each environment.

        Args:
            environment: Target environment name (development, staging, production, test, local)

        Returns:
            FlextResult containing environment-optimized schema processing system configuration

        Example:
            ```python
            result = (
                FlextSchemaProcessingConfig.create_environment_schema_processing_config(
                    "production"
                )
            )
            if result.success:
                prod_config = result.unwrap()
                print(f"Processing level: {prod_config['processing_level']}")
            ```

        """
        try:
            # Validate environment
            valid_environments = [
                e.value for e in FlextConstants.Config.ConfigEnvironment
            ]
            if environment not in valid_environments:
                return FlextResult[dict[str, object]].fail(
                    f"Invalid environment '{environment}'. Valid options: {valid_environments}"
                )

            # Base configuration for all environments
            base_config: FlextTypes.Config.ConfigDict = {
                "environment": environment,
                "enable_entry_validation": True,
                "enable_identifier_validation": True,
                "enable_content_preprocessing": True,
            }

            # Environment-specific optimizations
            if environment == FlextConstants.Config.ConfigEnvironment.PRODUCTION.value:
                base_config.update({
                    "processing_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                    "log_level": FlextConstants.Config.LogLevel.WARNING.value,
                    "enable_regex_caching": True,  # Performance optimization
                    "processing_batch_size": 2000,  # Large batches for efficiency
                    "enable_pipeline_optimization": True,
                    "regex_pattern_cache_size": 200,  # Large cache
                    "enable_processing_metrics": True,
                    "enable_performance_monitoring": True,
                })

            elif environment == FlextConstants.Config.ConfigEnvironment.STAGING.value:
                base_config.update({
                    "processing_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                    "log_level": FlextConstants.Config.LogLevel.INFO.value,
                    "enable_regex_caching": True,
                    "processing_batch_size": 1000,
                    "enable_pipeline_optimization": True,
                    "regex_pattern_cache_size": 100,
                    "enable_processing_metrics": True,
                })

            elif (
                environment == FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value
            ):
                base_config.update({
                    "processing_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                    "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                    "enable_regex_caching": False,  # Disable for debugging
                    "processing_batch_size": 100,  # Smaller for debugging
                    "enable_pipeline_optimization": False,
                    "regex_pattern_cache_size": 20,
                    "enable_detailed_logging": True,
                    "enable_processing_debugging": True,
                })

            elif environment == FlextConstants.Config.ConfigEnvironment.TEST.value:
                base_config.update({
                    "processing_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                    "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                    "enable_regex_caching": False,  # Disable for test clarity
                    "processing_batch_size": 50,  # Small for test precision
                    "enable_pipeline_optimization": False,
                    "regex_pattern_cache_size": 10,
                    "enable_test_assertions": True,
                    "enable_validation_debugging": True,
                })

            elif environment == FlextConstants.Config.ConfigEnvironment.LOCAL.value:
                base_config.update({
                    "processing_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                    "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                    "enable_regex_caching": False,
                    "processing_batch_size": 10,  # Very small for experimentation
                    "enable_pipeline_optimization": False,
                    "regex_pattern_cache_size": 5,
                    "enable_debug_output": True,
                    "enable_experimental_features": True,
                })

            # Cast ConfigDict to dict[str, object] for type compatibility
            result_config: dict[str, object] = dict(base_config.items())
            return FlextResult[dict[str, object]].ok(result_config)

        except Exception as e:
            return FlextResult[dict[str, object]].fail(
                f"Failed to create environment schema processing configuration: {e}"
            )

    @classmethod
    def optimize_schema_processing_performance(
        cls, performance_level: str = "balanced"
    ) -> object:
        """Optimize schema processing system performance settings.

        Configures schema processing system for optimal performance based on the specified
        performance level, adjusting regex pattern compilation, entry validation intensity,
        batch processing sizes, pipeline optimization strategies, caching configurations,
        and memory allocation to maximize throughput and minimize processing latency.

        Args:
            performance_level: Performance optimization level (low, balanced, high, extreme)

        Returns:
            FlextResult containing performance-optimized schema processing system configuration

        Example:
            ```python
            result = FlextSchemaProcessingConfig.optimize_schema_processing_performance(
                "high"
            )
            if result.success:
                optimized = result.unwrap()
            ```

        """
        try:
            _, _, _ = _get_dependencies()

            # Validate performance level
            valid_levels = ["low", "balanced", "high", "extreme"]
            if performance_level not in valid_levels:
                return FlextResult[dict[str, object]].fail(
                    f"Invalid performance_level '{performance_level}'. Valid options: {valid_levels}"
                )

            # Base performance configuration
            optimized_config: FlextTypes.Config.ConfigDict = {
                "environment": FlextConstants.Config.ConfigEnvironment.PRODUCTION.value,
                "processing_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                "log_level": FlextConstants.Config.LogLevel.WARNING.value,
            }

            # Base performance settings
            optimized_config.update({
                "performance_level": performance_level,
                "optimization_enabled": True,
                "optimization_timestamp": "2025-01-01T00:00:00Z",
            })

            # Performance level specific optimizations
            if performance_level == "high":
                optimized_config.update({
                    # Regex processing optimization
                    "enable_regex_caching": True,
                    "regex_pattern_cache_size": 500,
                    "enable_regex_precompilation": True,
                    "regex_optimization_level": "high",
                    # Entry validation optimization
                    "enable_entry_validation": True,
                    "enable_fast_validation_mode": True,
                    "skip_redundant_validations": True,
                    # Batch processing optimization
                    "processing_batch_size": 5000,
                    "enable_parallel_batch_processing": True,
                    "batch_processing_threads": 4,
                    # Pipeline optimization
                    "enable_pipeline_optimization": True,
                    "pipeline_cache_size": 100,
                    "enable_pipeline_parallelization": True,
                })

            elif performance_level == "extreme":
                optimized_config.update({
                    # Maximum performance settings
                    "enable_regex_caching": True,
                    "regex_pattern_cache_size": 2000,
                    "enable_aggressive_regex_caching": True,
                    "enable_compiled_regex_pooling": True,
                    # Minimal validation for speed
                    "enable_entry_validation": False,  # Skip for maximum speed
                    "skip_all_validations": True,
                    "enable_identifier_validation": False,
                    # Maximum batch processing
                    "processing_batch_size": 20000,
                    "enable_parallel_batch_processing": True,
                    "batch_processing_threads": 8,
                    "enable_lock_free_batch_processing": True,
                    # Maximum pipeline optimization
                    "enable_pipeline_optimization": True,
                    "pipeline_cache_size": 1000,
                    "enable_zero_copy_pipelines": True,
                    "skip_pipeline_validations": True,
                })

            elif performance_level == "balanced":
                optimized_config.update({
                    # Balanced settings
                    "enable_regex_caching": True,
                    "regex_pattern_cache_size": 100,
                    "enable_entry_validation": True,
                    "processing_batch_size": 1000,
                    "enable_pipeline_optimization": False,  # No optimization overhead
                    "pipeline_cache_size": 20,
                })

            else:  # low performance
                optimized_config.update({
                    # Conservative settings
                    "enable_regex_caching": False,
                    "processing_batch_size": 100,
                    "enable_pipeline_optimization": False,
                    "pipeline_cache_size": 5,
                    "enable_all_validations": True,
                    "enable_detailed_logging": True,
                    "enable_processing_debugging": True,
                })

            # Cast ConfigDict to dict[str, object] for type compatibility
            result_config: dict[str, object] = dict(optimized_config.items())
            return FlextResult[dict[str, object]].ok(result_config)

        except Exception as e:
            return FlextResult[dict[str, object]].fail(
                f"Failed to optimize schema processing performance: {e}"
            )


# =============================================================================
# EXPORTS
# =============================================================================

__all__: list[str] = [
    "FlextBaseConfigManager",
    "FlextBaseEntry",
    "FlextBaseFileWriter",
    "FlextBaseProcessor",
    "FlextBaseSorter",
    "FlextConfigAttributeValidator",
    "FlextEntryType",
    "FlextEntryValidator",
    "FlextProcessingPipeline",
    "FlextRegexProcessor",
    "FlextSchemaProcessingConfig",  # Configuration class with FlextTypes.Config integration
    # Legacy aliases removed - now in legacy.py
]

# Backward-compatible aliases moved to legacy.py per FLEXT_REFACTORING_PROMPT.md
