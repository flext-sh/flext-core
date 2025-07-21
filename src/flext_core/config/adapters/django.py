"""Django settings adapter.

Copyright (c) 2024 FLEXT Contributors
SPDX-License-Identifier: MIT

Base Django settings using Pydantic.

Provides type-safe Django settings with environment variable support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import Field
from pydantic import field_validator
from pydantic_settings import SettingsConfigDict

from flext_core.config.base import BaseSettings
from flext_core.config.validators import validate_database_url

# Development defaults - change in production
DEV_SECRET_KEY = "development-key-change-in-production"


class DjangoSettings(BaseSettings):
    """Django settings."""

    model_config = SettingsConfigDict(env_prefix="FLEXT_")

    # Core Django settings
    secret_key: str = Field(..., description="Django secret key")
    debug: bool = Field(default=False, description="Debug mode")
    allowed_hosts: list[str] = Field(
        default_factory=lambda: ["localhost", "127.0.0.1"],
        description="Allowed hosts",
    )

    # Database
    database_url: str = Field(
        default="sqlite:///db.sqlite3",
        description="Database connection URL",
    )

    # Static files
    static_url: str = Field(default="/static/", description="Static files URL")
    static_root: Path | None = Field(default=None, description="Static files root")
    media_url: str = Field(default="/media/", description="Media files URL")
    media_root: Path | None = Field(default=None, description="Media files root")

    # Security
    secure_ssl_redirect: bool = Field(default=False, description="Force HTTPS")
    session_cookie_secure: bool = Field(
        default=False,
        description="Secure session cookies",
    )
    csrf_cookie_secure: bool = Field(default=False, description="Secure CSRF cookies")
    secure_hsts_seconds: int = Field(default=0, description="HSTS max age")

    # Email
    email_backend: str = Field(
        default="django.core.mail.backends.console.EmailBackend",
        description="Email backend",
    )
    email_host: str | None = Field(default=None, description="Email host")
    email_port: int = Field(default=587, description="Email port")
    email_use_tls: bool = Field(default=True, description="Use TLS for email")
    email_host_user: str | None = Field(default=None, description="Email username")
    email_host_password: str | None = Field(default=None, description="Email password")

    # Cache
    cache_backend: str = Field(
        default="django.core.cache.backends.locmem.LocMemCache",
        description="Cache backend",
    )
    cache_location: str | None = Field(default=None, description="Cache location")

    # Internationalization
    language_code: str = Field(default="en-us", description="Language code")
    time_zone: str = Field(default="UTC", description="Time zone")
    use_i18n: bool = Field(default=True, description="Enable internationalization")
    use_tz: bool = Field(default=True, description="Use timezone-aware datetimes")

    @field_validator("database_url")
    @classmethod
    def validate_database_url_field(cls, v: str) -> str:
        """Validate database URL format.

        Arguments:
            v: The database URL to validate.

        Returns:
            The validated database URL.

        """
        return validate_database_url(v)

    @field_validator("allowed_hosts")
    @classmethod
    def validate_allowed_hosts(cls, v: list[str]) -> list[str]:
        """Validate allowed hosts are not empty in production.

        Arguments:
            v: The allowed hosts to validate.

        Raises:
            ValueError: If the allowed hosts are empty in production.

        Returns:
            The validated allowed hosts.

        """
        debug_field = cls.model_fields.get("debug")
        debug_default = debug_field.default if debug_field else False
        if not v and not debug_default:
            msg = "ALLOWED_HOSTS is required in production mode"
            raise ValueError(msg)

        return v

    def to_django_settings(self) -> dict[str, Any]:
        """Convert to Django settings dictionary format.

        Returns:
            The Django settings dictionary.

        """
        settings = self.model_dump()

        # Convert database URL to Django database dict
        db_url = urlparse(settings.pop("database_url"))

        # Map URL schemes to Django database engines
        engine_map = {
            "postgresql": "django.db.backends.postgresql",
            "postgres": "django.db.backends.postgresql",
            "mysql": "django.db.backends.mysql",
            "sqlite": "django.db.backends.sqlite3",
            "oracle": "django.db.backends.oracle",
        }

        databases = {
            "default": {
                "ENGINE": engine_map.get(db_url.scheme, db_url.scheme),
                "NAME": db_url.path[1:] if db_url.path else "",
                "USER": db_url.username or "",
                "PASSWORD": db_url.password or "",
                "HOST": db_url.hostname or "",
                "PORT": db_url.port or "",
            },
        }

        # Handle SQLite special case
        if db_url.scheme == "sqlite":
            databases["default"]["NAME"] = db_url.path

        settings["DATABASES"] = databases

        # Convert field names to Django conventions
        django_settings = {}
        for key, value in settings.items():
            django_key = key.upper()
            django_settings[django_key] = value

        # Add computed settings
        if settings.get("static_root") is None:
            django_settings["STATIC_ROOT"] = Path("staticfiles")

        if settings.get("media_root") is None:
            django_settings["MEDIA_ROOT"] = Path("media")

        return django_settings


def django_settings_adapter(
    pydantic_settings: DjangoSettings | type[DjangoSettings],
    additional_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert pydantic settings to Django settings dictionary.

    Args:
        pydantic_settings: Django settings instance or class
        additional_settings: Additional settings to merge

    Returns:
        Dictionary compatible with Django settings

    """
    # Get settings instance
    if isinstance(pydantic_settings, type):
        settings_instance = pydantic_settings(
            project_name="django-app",
            project_version="1.0.0",
            environment="development",
            secret_key=DEV_SECRET_KEY,
        )
    else:
        settings_instance = pydantic_settings

    # Convert to Django format
    django_settings = settings_instance.to_django_settings()

    # Add standard Django settings
    django_settings.update(
        {
            # Apps (to be extended by specific projects)
            "INSTALLED_APPS": [
                "django.contrib.REDACTED_LDAP_BIND_PASSWORD",
                "django.contrib.auth",
                "django.contrib.contenttypes",
                "django.contrib.sessions",
                "django.contrib.messages",
                "django.contrib.staticfiles",
            ],
            # Middleware
            "MIDDLEWARE": [
                "django.middleware.security.SecurityMiddleware",
                "django.contrib.sessions.middleware.SessionMiddleware",
                "django.middleware.common.CommonMiddleware",
                "django.middleware.csrf.CsrfViewMiddleware",
                "django.contrib.auth.middleware.AuthenticationMiddleware",
                "django.contrib.messages.middleware.MessageMiddleware",
                "django.middleware.clickjacking.XFrameOptionsMiddleware",
            ],
            # Templates
            "TEMPLATES": [
                {
                    "BACKEND": "django.template.backends.django.DjangoTemplates",
                    "DIRS": [],
                    "APP_DIRS": True,
                    "OPTIONS": {
                        "context_processors": [
                            "django.template.context_processors.debug",
                            "django.template.context_processors.request",
                            "django.contrib.auth.context_processors.auth",
                            "django.contrib.messages.context_processors.messages",
                        ],
                    },
                },
            ],
            # Password validators
            "AUTH_PASSWORD_VALIDATORS": [
                {
                    "NAME": (
                        "django.contrib.auth.password_validation"
                        ".UserAttributeSimilarityValidator"
                    ),
                },
                {
                    "NAME": (
                        "django.contrib.auth.password_validation.MinimumLengthValidator"
                    ),
                },
                {
                    "NAME": (
                        "django.contrib.auth.password_validation"
                        ".CommonPasswordValidator"
                    ),
                },
                {
                    "NAME": (
                        "django.contrib.auth.password_validation"
                        ".NumericPasswordValidator"
                    ),
                },
            ],
            # Other settings
            "ROOT_URLCONF": "urls",
            "WSGI_APPLICATION": "wsgi.application",
            "DEFAULT_AUTO_FIELD": "django.db.models.BigAutoField",
        },
    )

    # Merge additional settings
    if additional_settings:
        django_settings.update(additional_settings)

    return django_settings


def create_django_settings_module(
    settings_class: type[DjangoSettings],
    module_name: str = "settings",
    base_dir: Path | None = None,
) -> None:
    """Create a Django settings module from FLEXT configuration.

    Args:
        settings_class: Django settings class to use
        module_name: Name of the settings module
        base_dir: Base directory for the project

    """
    if base_dir is None:
        base_dir = Path.cwd()

    # Create settings instance
    settings = settings_class(
        project_name="django-app",
        project_version="1.0.0",
        environment="development",
        secret_key=DEV_SECRET_KEY,
    )

    # Get Django settings
    django_settings = django_settings_adapter(settings)

    # Add BASE_DIR
    django_settings["BASE_DIR"] = base_dir

    # Generate settings file content
    content = '''"""
Django settings generated from FLEXT configuration.
"""

from pathlib import Path

# Build paths inside the project
BASE_DIR = Path(__file__).resolve().parent.parent

'''

    # Add settings
    for key, value in django_settings.items():
        if isinstance(value, str | Path):
            content += f'{key} = "{value}"\n'
        else:
            content += f"{key} = {value!r}\n"

    # Write settings file
    settings_file = base_dir / f"{module_name}.py"
    settings_file.write_text(content)


__all__ = [
    "DjangoSettings",
    "create_django_settings_module",
    "django_settings_adapter",
]
