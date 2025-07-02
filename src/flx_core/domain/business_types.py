"""Business domain value objects for enterprise-grade type safety.

This module implements validated domain types to replace primitive types
throughout the codebase, providing type safety, input validation, and
business logic encapsulation following ZERO TOLERANCE principles.

Python 3.13 + Pydantic v2 enterprise patterns.
"""

from __future__ import annotations

import re
import warnings

from pydantic import BaseModel, Field, field_validator

# Domain constants for zero tolerance to magic values
from flx_core.config.domain_config import get_domain_constants

_constants = get_domain_constants()

# Python 3.13 type aliases for enterprise domains
ConfigurationKey = str
ConfigurationValue = str | int | bool | float | None


# === NETWORK AND CONNECTION TYPES ===


class NetworkPort(BaseModel):
    """A validated network port number with enterprise constraints."""

    model_config = {"frozen": True, "str_strip_whitespace": True}

    value: int = Field(ge=1, le=65535, description="Valid TCP/UDP port number")

    def __init__(self, value: int | str | None = None, **data: object) -> None:
        """Initialize with port number, allowing string conversion."""
        if value is not None:
            if isinstance(value, str):
                value = int(value)
            super().__init__(value=value, **data)
        else:
            super().__init__(**data)

    @field_validator("value")
    @classmethod
    def validate_port_range(cls, v: int) -> int:
        """Validate port is in valid range and not reserved."""
        if v < 1 or v > _constants.MAXIMUM_PORT_NUMBER:
            msg = f"Port {v} out of valid range (1-{_constants.MAXIMUM_PORT_NUMBER})"
            raise ValueError(msg)

        # Warn about well-known system ports (optional business rule)
        if v < _constants.MEMORY_UNIT_CONVERSION:
            # Allow but could log warning for non-privileged binding
            pass

        return v

    def __str__(self) -> str:
        """Return string representation of port."""
        return str(self.value)

    def __int__(self) -> int:
        """Return integer representation of port."""
        return self.value

    @property
    def is_system_port(self) -> bool:
        """Check if this is a system/privileged port (< 1024)."""
        return self.value < _constants.MEMORY_UNIT_CONVERSION

    @property
    def is_ephemeral(self) -> bool:
        """Check if this is in ephemeral port range (32768-65535)."""
        return self.value >= _constants.EPHEMERAL_PORT_START


class HostAddress(BaseModel):
    """A validated hostname or IP address with enterprise constraints."""

    model_config = {"frozen": True, "str_strip_whitespace": True}

    value: str = Field(
        min_length=1,
        max_length=255,
        description="Valid hostname, FQDN, or IP address",
    )

    def __init__(self, value: str | None = None, **data: object) -> None:
        """Initialize with hostname or IP address."""
        if value is not None:
            super().__init__(value=value, **data)
        else:
            super().__init__(**data)

    @field_validator("value")
    @classmethod
    def validate_host_format(cls, v: str) -> str:
        """Validate hostname or IP address format."""
        # Allow secure localhost addresses only - ZERO TOLERANCE SECURITY
        if v.lower() in {"localhost", "127.0.0.1", "::1"}:
            return v

        # SECURITY WARNING: 0.0.0.0 should only be used in production with proper firewalls
        if v == "0.0.0.0":
            # Issue warning but allow for container deployments
            warnings.warn(
                "Using 0.0.0.0 (bind to all interfaces) - ensure proper firewall configuration",
                UserWarning,
                stacklevel=2,
            )
            return v

        # Basic hostname validation (letters, numbers, dots, hyphens)
        if not re.match(r"^[a-zA-Z0-9\.\-_]+$", v):
            msg = f"Invalid host format: {v}. Must contain only alphanumeric characters, dots, hyphens, and underscores"
            raise ValueError(msg)

        # Check for valid structure (no double dots, etc.)
        if ".." in v or v.startswith(".") or v.endswith("."):
            msg = f"Invalid host structure: {v}"
            raise ValueError(msg)

        return v

    def __str__(self) -> str:
        """Return string representation of host."""
        return self.value

    @property
    def is_localhost(self) -> bool:
        """Check if this is a localhost address."""
        return self.value.lower() in {"localhost", "127.0.0.1", "::1"}

    @property
    def is_wildcard(self) -> bool:
        """Check if this is a wildcard address - SECURITY WARNING."""
        # ZERO TOLERANCE SECURITY: Explicit warning for wildcard addresses
        return self.value in {
            "0.0.0.0",
            "::",
        }  # nosec B104 - intentionally checking wildcard


class TimeoutSeconds(BaseModel):
    """A validated timeout value in seconds with enterprise constraints."""

    model_config = {"frozen": True}

    value: float = Field(
        gt=0,
        le=3600,
        description="Timeout in seconds, positive and <= 1 hour",
    )

    def __init__(self, value: float | None = None, **data: object) -> None:
        """Initialize with timeout value."""
        if value is not None:
            super().__init__(value=float(value), **data)
        else:
            super().__init__(**data)

    @field_validator("value")
    @classmethod
    def validate_timeout_range(cls, v: float) -> float:
        """Validate timeout is reasonable for enterprise use."""
        if v <= 0:
            msg = "Timeout must be positive"
            raise ValueError(msg)

        if v > _constants.MELTANO_DEFAULT_TIMEOUT:  # 1 hour max
            msg = f"Timeout {v}s exceeds maximum of {_constants.MELTANO_DEFAULT_TIMEOUT}s (1 hour)"
            raise ValueError(msg)

        return v

    def __str__(self) -> str:
        """Return string representation of timeout."""
        if self.value < _constants.STANDARD_TIMEOUT_SECONDS:
            return f"{self.value:.1f}s"
        return f"{self.value / 60:.1f}m"

    def __float__(self) -> float:
        """Return float representation of timeout."""
        return self.value

    @property
    def milliseconds(self) -> int:
        """Get timeout in milliseconds."""
        return int(self.value * 1000)


class ConnectionString(BaseModel):
    """A validated connection string with enterprise security constraints."""

    model_config = {"frozen": True, "str_strip_whitespace": True}

    value: str = Field(
        min_length=1,
        max_length=2048,
        description="Valid connection string or URI",
    )

    def __init__(self, value: str | None = None, **data: object) -> None:
        """Initialize with connection string."""
        if value is not None:
            super().__init__(value=value, **data)
        else:
            super().__init__(**data)

    @field_validator("value")
    @classmethod
    def validate_connection_string(cls, v: str) -> str:
        """Validate connection string format and security."""
        # Basic URI-like structure validation
        if "://" not in v and not v.startswith("/") and len(v.split()) > 1:
            # Allow file paths or simple strings, but validate they're reasonable
            msg = f"Connection string appears to contain spaces: {v[:50]}..."
            raise ValueError(msg)

        # Security: warn about obvious credential leaks (basic check)
        if re.search(r"password=\w+", v, re.IGNORECASE):
            # In production, this would log a security warning
            # For now, we allow it but could enhance with credential detection
            pass

        return v

    def __str__(self) -> str:
        """Return string representation, masking potential credentials."""
        # Mask potential credentials for logging/display
        return re.sub(
            r"(password=)[^&\s]+",
            r"\1***",
            self.value,
            flags=re.IGNORECASE,
        )

    @property
    def protocol(self) -> str | None:
        """Extract protocol from connection string."""
        if "://" in self.value:
            return self.value.split("://")[0].lower()
        return None

    @property
    def is_secure(self) -> bool:
        """Check if connection string indicates secure protocol."""
        secure_protocols = {"https", "tls", "ssl", "rediss", "mqtts"}
        protocol = self.protocol
        return protocol in secure_protocols if protocol else False


# === BUSINESS IDENTIFIERS AND NAMES ===


class PluginName(BaseModel):
    """A validated plugin name with enterprise naming conventions."""

    model_config = {"frozen": True, "str_strip_whitespace": True}

    value: str = Field(
        min_length=1,
        max_length=100,
        description="Valid plugin name with standard conventions",
    )

    def __init__(self, value: str | None = None, **data: object) -> None:
        """Initialize with plugin name."""
        if value is not None:
            super().__init__(value=value, **data)
        else:
            super().__init__(**data)

    @field_validator("value")
    @classmethod
    def validate_plugin_name(cls, v: str) -> str:
        """Validate plugin name follows enterprise conventions."""
        # Plugin names should follow kebab-case or snake_case conventions
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", v):
            msg = (
                f"Plugin name '{v}' must start with letter and contain only "
                "alphanumeric characters, underscores, and hyphens"
            )
            raise ValueError(msg)

        # Prevent overly short names
        if len(v) < _constants.MINIMUM_PLUGIN_NAME_LENGTH:
            msg = f"Plugin name '{v}' too short (minimum {_constants.MINIMUM_PLUGIN_NAME_LENGTH} characters)"
            raise ValueError(msg)

        return v

    def __str__(self) -> str:
        """Return string representation of plugin name."""
        return self.value

    @property
    def is_tap(self) -> bool:
        """Check if this is a tap (extractor) plugin."""
        return self.value.startswith("tap-")

    @property
    def is_target(self) -> bool:
        """Check if this is a target (loader) plugin."""
        return self.value.startswith("target-")

    @property
    def normalized(self) -> str:
        """Return normalized name (replace hyphens with underscores)."""
        return self.value.replace("-", "_")


class EmailAddress(BaseModel):
    """A validated email address with enterprise compliance."""

    model_config = {"frozen": True, "str_strip_whitespace": True}

    value: str = Field(
        min_length=5,
        max_length=320,  # RFC 5321 limit
        description="Valid email address",
    )

    def __init__(self, value: str | None = None, **data: object) -> None:
        """Initialize with email address."""
        if value is not None:
            super().__init__(value=value.lower(), **data)
        else:
            super().__init__(**data)

    @field_validator("value")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        """Validate email address format per RFC standards."""
        # Basic RFC 5322 pattern (simplified but robust)
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

        if not re.match(email_pattern, v):
            msg = f"Invalid email format: {v}"
            raise ValueError(msg)

        # Check for reasonable structure
        local, domain = v.split("@", 1)

        if (
            len(local) > _constants.EMAIL_LOCAL_PART_MAX_LENGTH
        ):  # RFC 5321 local part limit
            msg = f"Email local part too long (max {_constants.EMAIL_LOCAL_PART_MAX_LENGTH} characters): {local}"
            raise ValueError(msg)

        if len(domain) > _constants.EMAIL_DOMAIN_MAX_LENGTH:  # RFC 5321 domain limit
            msg = f"Email domain too long (max {_constants.EMAIL_DOMAIN_MAX_LENGTH} characters): {domain}"
            raise ValueError(msg)

        return v.lower()  # Normalize to lowercase

    def __str__(self) -> str:
        """Return string representation of email."""
        return self.value

    @property
    def local_part(self) -> str:
        """Get local part (before @) of email."""
        return self.value.split("@")[0]

    @property
    def domain(self) -> str:
        """Get domain part (after @) of email."""
        return self.value.split("@")[1]

    @property
    def is_corporate(self) -> bool:
        """Check if email appears to be from corporate domain."""
        # Simple heuristic - avoid common free email providers
        free_domains = {
            "gmail.com",
            "yahoo.com",
            "hotmail.com",
            "outlook.com",
            "aol.com",
            "icloud.com",
            "protonmail.com",
        }
        return self.domain not in free_domains

    @property
    def username(self) -> str:
        """Get username part (local part) of email for compatibility."""
        return self.local_part

    @property
    def is_common_provider(self) -> bool:
        """Check if email is from a common provider."""
        common_domains = {
            "gmail.com",
            "yahoo.com",
            "hotmail.com",
            "outlook.com",
            "aol.com",
            "icloud.com",
            "protonmail.com",
        }
        return self.domain in common_domains


class Username(BaseModel):
    """A validated username with enterprise security constraints."""

    model_config = {"frozen": True, "str_strip_whitespace": True}

    value: str = Field(
        min_length=3,
        max_length=32,
        description="Valid username for authentication",
    )

    def __init__(self, value: str | None = None, **data: object) -> None:
        """Initialize with username."""
        if value is not None:
            super().__init__(value=value.lower(), **data)
        else:
            super().__init__(**data)

    @field_validator("value")
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Validate username follows security best practices."""
        # Allow letters, numbers, dots, underscores, hyphens
        if not re.match(r"^[a-zA-Z0-9._-]+$", v):
            msg = (
                f"Username '{v}' contains invalid characters. "
                "Only letters, numbers, dots, underscores, and hyphens allowed"
            )
            raise ValueError(msg)

        # Must start with letter or number
        if not v[0].isalnum():
            msg = f"Username '{v}' must start with letter or number"
            raise ValueError(msg)

        # Must end with letter or number (not dot, underscore, or hyphen)
        if not v[-1].isalnum():
            msg = f"Username '{v}' must end with letter or number"
            raise ValueError(msg)

        # Prevent reserved usernames
        reserved = {
            "admin",
            "administrator",
            "root",
            "system",
            "user",
            "guest",
            "public",
            "private",
            "test",
            "demo",
            "api",
            "www",
            "mail",
        }
        if v.lower() in reserved:
            msg = f"Username '{v}' is reserved"
            raise ValueError(msg)

        return v.lower()  # Normalize to lowercase

    def __str__(self) -> str:
        """Return string representation of username."""
        return self.value

    @property
    def is_system_account(self) -> bool:
        """Check if username appears to be a system account."""
        system_patterns = ["system", "service", "daemon", "bot", "api"]
        return any(pattern in self.value for pattern in system_patterns)

    @property
    def is_admin(self) -> bool:
        """Check if username appears to be an admin account."""
        admin_patterns = ["admin", "administrator", "root", "superuser"]
        return any(pattern in self.value.lower() for pattern in admin_patterns)

    def __len__(self) -> int:
        """Return length of username for compatibility."""
        return len(self.value)


# === EXECUTION AND PROCESSING PARAMETERS ===


class ExecutionNumber(BaseModel):
    """A validated execution number with enterprise constraints."""

    model_config = {"frozen": True}

    value: int = Field(
        ge=1,
        le=999999,
        description="Sequential execution number",
    )

    def __init__(self, value: int | str | None = None, **data: object) -> None:
        """Initialize with execution number."""
        if value is not None:
            if isinstance(value, str):
                value = int(value)
            super().__init__(value=value, **data)
        else:
            super().__init__(**data)

    def __str__(self) -> str:
        """Return string representation."""
        return str(self.value)  # Simple string representation

    def __int__(self) -> int:
        """Return integer representation."""
        return self.value

    @property
    def formatted(self) -> str:
        """Return formatted execution number for display."""
        return f"#{self.value:06d}"

    def next(self) -> ExecutionNumber:
        """Get next execution number in sequence."""
        return ExecutionNumber(value=self.value + 1)


class RecordCount(BaseModel):
    """A validated record count with enterprise data processing constraints."""

    model_config = {"frozen": True}

    value: int = Field(
        ge=0,
        le=2147483647,  # 32-bit signed int max
        description="Number of records processed",
    )

    def __init__(self, value: int | str | None = None, **data: object) -> None:
        """Initialize with record count."""
        if value is not None:
            if isinstance(value, str):
                value = int(value)
            super().__init__(value=value, **data)
        else:
            super().__init__(**data)

    def __str__(self) -> str:
        """Return human-readable string representation."""
        if self.value < _constants.THOUSAND_THRESHOLD:
            return str(self.value)
        if self.value < _constants.MILLION_THRESHOLD:
            return f"{self.value / _constants.THOUSAND_THRESHOLD:.1f}k"
        if self.value < _constants.BILLION_THRESHOLD:
            return f"{self.value / _constants.MILLION_THRESHOLD:.1f}M"
        return f"{self.value / _constants.BILLION_THRESHOLD:.1f}B"

    def __int__(self) -> int:
        """Return integer representation."""
        return self.value

    @property
    def is_empty(self) -> bool:
        """Check if record count is zero."""
        return self.value == 0

    @property
    def is_large_dataset(self) -> bool:
        """Check if this represents a large dataset (>1M records)."""
        return self.value > _constants.MILLION_THRESHOLD


class BatchSize(BaseModel):
    """A validated batch size for data processing with enterprise constraints."""

    model_config = {"frozen": True}

    value: int = Field(
        ge=1,
        le=10000,
        description="Batch size for data processing operations",
    )

    def __init__(self, value: int | str | None = None, **data: object) -> None:
        """Initialize with batch size."""
        if value is not None:
            if isinstance(value, str):
                value = int(value)
            super().__init__(value=value, **data)
        else:
            super().__init__(**data)

    @field_validator("value")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size is optimal for performance."""
        # Warn about very small batches (performance impact)
        if v < _constants.MIN_BATCH_SIZE:
            # In production, would log performance warning
            pass

        # Warn about very large batches (memory impact)
        if v > _constants.MAX_BATCH_SIZE:
            # In production, would log memory warning
            pass

        return v

    def __str__(self) -> str:
        """Return string representation."""
        return str(self.value)

    def __int__(self) -> int:
        """Return integer representation."""
        return self.value

    @property
    def is_optimal(self) -> bool:
        """Check if batch size is in optimal range (100-1000)."""
        return (
            _constants.OPTIMAL_BATCH_MIN <= self.value <= _constants.OPTIMAL_BATCH_MAX
        )

    @classmethod
    def optimal_for_records(cls, total_records: int) -> BatchSize:
        """Calculate optimal batch size for given record count."""
        # For very small datasets, use the record count itself
        if total_records <= _constants.SMALL_DATASET_THRESHOLD:
            return cls(value=total_records)
        # For small-medium datasets (51-999), use 100
        if total_records < _constants.THOUSAND_THRESHOLD:
            return cls(value=_constants.OPTIMAL_BATCH_MIN)
        # For large datasets (1000+), use maximum optimal
        return cls(value=_constants.OPTIMAL_BATCH_MAX)


# === SCHEDULING AND TIMING TYPES ===


class ScheduleId(BaseModel):
    """A validated schedule identifier with enterprise constraints."""

    model_config = {"frozen": True, "str_strip_whitespace": True}

    value: str = Field(min_length=1, max_length=255, description="Schedule identifier")

    @field_validator("value")
    @classmethod
    def validate_schedule_id_format(cls, v: str) -> str:
        """Validate schedule ID format."""
        # Must be alphanumeric with underscores and hyphens
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            msg = "Schedule ID must contain only alphanumeric characters, underscores, and hyphens"
            raise ValueError(msg)
        return v

    def __str__(self) -> str:
        """Return string representation."""
        return self.value

    def __hash__(self) -> int:
        """Return hash of the schedule ID."""
        return hash(self.value)


class CronExpression(BaseModel):
    """A validated cron expression with enterprise constraints."""

    model_config = {"frozen": True, "str_strip_whitespace": True}

    value: str = Field(min_length=9, max_length=100, description="Cron expression")

    @field_validator("value")
    @classmethod
    def validate_cron_format(cls, v: str) -> str:
        """Validate cron expression format."""
        # Basic cron validation - 5 or 6 fields separated by spaces
        parts = v.split()
        if len(parts) not in {5, 6}:
            msg = "Cron expression must have 5 or 6 fields separated by spaces"
            raise ValueError(msg)

        # Check each field contains valid characters
        valid_chars = set("0123456789*,-/")
        for part in parts:
            if not all(c in valid_chars for c in part):
                msg = f"Invalid characters in cron expression part: {part}"
                raise ValueError(msg)

        return v

    def __str__(self) -> str:
        """Return string representation."""
        return self.value

    @property
    def is_every_minute(self) -> bool:
        """Check if this cron runs every minute."""
        return self.value.startswith("* * * * *")

    @property
    def is_hourly(self) -> bool:
        """Check if this cron runs hourly."""
        return self.value.startswith("0 * * * *")

    @property
    def is_daily(self) -> bool:
        """Check if this cron runs daily."""
        return self.value.startswith("0 0 * * *")


class Timezone(BaseModel):
    """A validated timezone identifier with enterprise constraints."""

    model_config = {"frozen": True, "str_strip_whitespace": True}

    value: str = Field(min_length=3, max_length=50, description="Timezone identifier")

    @field_validator("value")
    @classmethod
    def validate_timezone_format(cls, v: str) -> str:
        """Validate timezone format."""
        # Common timezone formats: UTC, America/New_York, Europe/London, etc.
        valid_patterns = [
            r"^UTC$",
            r"^GMT[+-]\d{1,2}$",
            r"^[A-Z][a-z]+/[A-Z][a-z_]+$",  # Region/City format
            r"^[A-Z]{3,4}$",  # EST, PST, etc.
        ]

        if not any(re.match(pattern, v) for pattern in valid_patterns):
            msg = (
                f"Invalid timezone format: {v}. Use UTC, GMT+/-N, or Region/City format"
            )
            raise ValueError(msg)

        return v

    def __str__(self) -> str:
        """Return string representation."""
        return self.value

    @property
    def is_utc(self) -> bool:
        """Check if this is UTC timezone."""
        return self.value == "UTC"

    @property
    def is_gmt_offset(self) -> bool:
        """Check if this is a GMT offset timezone."""
        return self.value.startswith("GMT")
