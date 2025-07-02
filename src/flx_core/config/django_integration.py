"""Django Integration for Unified Domain Configuration.

This module provides seamless integration between the unified domain configuration
and Django settings, eliminating configuration duplication and ensuring consistency.

ZERO TOLERANCE: No duplicate Django configuration anywhere in the project.
"""

from __future__ import annotations

import re
from typing import Any

from flx_core.config.domain_config import get_config, get_domain_constants


def get_django_database_config() -> dict[str, Any]:
    """Get Django-compatible database configuration from domain config."""
    config = get_config()
    database_url = config.database.url

    # Parse database URL to Django format
    if database_url.startswith("sqlite://"):
        return {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": database_url.replace("sqlite:///", ""),
            "OPTIONS": {
                "timeout": config.database.query_timeout,
            },
        }
    if database_url.startswith("postgresql://"):
        # Parse postgresql://user:password@host:port/dbname
        match = re.match(
            r"postgresql://(?:([^:@]+)(?::([^@]+))?@)?([^:]+)(?::(\d+))?/(.+)",
            database_url,
        )
        if match:
            user, password, host, port, dbname = match.groups()
            return {
                "ENGINE": "django.db.backends.postgresql",
                "NAME": dbname,
                "USER": user or "",
                "PASSWORD": password or "",
                "HOST": host,
                "PORT": port or "5432",
                "OPTIONS": {
                    "connect_timeout": int(config.database.pool_timeout),
                },
            }

    # ZERO TOLERANCE - Invalid database URL format
    msg = f"Invalid database URL format: {database_url}"
    raise ValueError(msg)


def get_django_cache_config() -> dict[str, Any]:
    """Get Django-compatible cache configuration from domain config."""
    config = get_config()

    return {
        "default": {
            "BACKEND": "django_redis.cache.RedisCache",
            "LOCATION": f"redis://localhost:{config.network.redis_port}/0",
            "OPTIONS": {
                "CLIENT_CLASS": "django_redis.client.DefaultClient",
                "CONNECTION_POOL_KWARGS": {
                    "max_connections": min(config.network.max_connections, 50),
                },
            },
        },
    }


def get_django_logging_config() -> dict[str, Any]:
    """Get Django-compatible logging configuration from domain config."""
    config = get_config()

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "verbose": {
                "format": config.monitoring.log_format,
            },
            "simple": {
                "format": "{levelname} {message}",
                "style": "{",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "verbose",
            },
        },
        "root": {
            "handlers": ["console"],
            "level": config.monitoring.log_level,
        },
        "loggers": {
            "django": {
                "handlers": ["console"],
                "level": config.monitoring.log_level,
                "propagate": False,
            },
            "flx_web": {
                "handlers": ["console"],
                "level": "DEBUG" if config.debug else config.monitoring.log_level,
                "propagate": False,
            },
        },
    }


def get_django_cors_config() -> dict[str, Any]:
    """Get Django-compatible CORS configuration from domain config."""
    config = get_config()
    api_config = config.get_service_config("api")

    return {
        "CORS_ALLOWED_ORIGINS": api_config.get(
            "cors_origins",
            [
                "http://localhost:3000",
                "http://localhost:8080",
                f"http://localhost:{config.network.web_port}",
            ],
        ),
        "CORS_ALLOW_ALL_ORIGINS": config.is_development,
        "CORS_ALLOW_CREDENTIALS": True,
    }


def get_django_security_config() -> dict[str, Any]:
    """Get Django-compatible security configuration from domain config."""
    config = get_config()

    return {
        "USE_TZ": True,
        "SECURE_BROWSER_XSS_FILTER": True,
        "SECURE_CONTENT_TYPE_NOSNIFF": True,
        "X_FRAME_OPTIONS": "DENY",
        "SECURE_HSTS_SECONDS": (
            get_domain_constants().HSTS_MAX_AGE_SECONDS if config.is_production else 0
        ),
        "SECURE_HSTS_INCLUDE_SUBDOMAINS": config.is_production,
        "SECURE_HSTS_PRELOAD": config.is_production,
        "SECURE_SSL_REDIRECT": config.is_production,
        "SESSION_COOKIE_SECURE": config.is_production,
        "CSRF_COOKIE_SECURE": config.is_production,
        "SESSION_COOKIE_AGE": config.security.session_timeout_minutes * 60,
    }


def get_complete_django_settings() -> dict[str, Any]:
    """Get complete Django settings dictionary from unified domain configuration.

    This replaces ALL Django settings with configuration from domain_config.
    with strict validation
    """
    config = get_config()
    django_base = config.get_django_settings_dict()

    # Enhance base settings with additional configurations
    return {
        **django_base,
        # Database configuration
        "DATABASES": {
            "default": get_django_database_config(),
        },
        # Cache configuration
        "CACHES": get_django_cache_config(),
        # Logging configuration
        "LOGGING": get_django_logging_config(),
        # Security configuration
        **get_django_security_config(),
        # CORS configuration
        **get_django_cors_config(),
        # Application-specific settings from domain config
        "FLX_GRPC_HOST": "localhost",
        "FLX_GRPC_PORT": config.network.grpc_port,
        "FLX_API_PORT": config.network.api_port,
        "FLX_WEB_PORT": config.network.web_port,
        "FLX_WEBSOCKET_PORT": config.network.websocket_port,
        # Internationalization
        "LANGUAGE_CODE": "en-us",
        "TIME_ZONE": "UTC",
        "USE_I18N": True,
        "USE_TZ": True,
        # Media and static files (keep defaults for now)
        "STATIC_URL": "/static/",
        "MEDIA_URL": "/media/",
        # Default primary key field type
        "DEFAULT_AUTO_FIELD": "django.db.models.BigAutoField",
    }


def inject_domain_config_into_django_settings(settings_dict: dict[str, Any]) -> None:
    """Inject domain configuration into existing Django settings dictionary.

    This modifies the settings dictionary in-place, replacing key settings
    with values from the unified domain configuration.
    """
    domain_settings = get_complete_django_settings()

    # Update key settings while preserving Django-specific configurations
    settings_dict.update(dict(domain_settings.items()))


# Export utility functions for direct use in Django settings files
__all__ = [
    "get_complete_django_settings",
    "get_django_cache_config",
    "get_django_cors_config",
    "get_django_database_config",
    "get_django_logging_config",
    "get_django_security_config",
    "inject_domain_config_into_django_settings",
]
