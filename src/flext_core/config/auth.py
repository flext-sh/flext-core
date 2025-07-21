"""Authentication configuration patterns - consolidated from multiple projects.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module consolidates authentication configuration patterns found across:
- flext-auth, flext-api, flext-grpc, flext-web
"""

from __future__ import annotations

from pydantic import Field
from pydantic import field_validator

from flext_core.config.base import BaseConfig


class JWTConfig(BaseConfig):
    """JWT authentication configuration - consolidated pattern."""

    secret_key: str = Field(
        ...,
        description="JWT secret key",
        json_schema_extra={"secret": True},
    )
    algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    access_token_expire_minutes: int = Field(
        default=30,
        ge=1,
        le=1440,
        description="Access token expiration in minutes",
    )
    refresh_token_expire_days: int = Field(
        default=7,
        ge=1,
        le=30,
        description="Refresh token expiration in days",
    )
    issuer: str = Field(default="flext", description="JWT issuer")
    audience: str = Field(default="flext", description="JWT audience")

    @field_validator("algorithm")
    @classmethod
    def validate_algorithm(cls, v: str) -> str:
        """Validate JWT algorithm."""
        allowed_algorithms = {
            "HS256",
            "HS384",
            "HS512",  # HMAC
            "RS256",
            "RS384",
            "RS512",  # RSA
            "ES256",
            "ES384",
            "ES512",  # ECDSA
        }
        if v not in allowed_algorithms:
            msg = f"Algorithm must be one of: {allowed_algorithms}"
            raise ValueError(msg)
        return v


class OAuth2Config(BaseConfig):
    """OAuth2 configuration - consolidated pattern."""

    client_id: str = Field(..., description="OAuth2 client ID")
    client_secret: str = Field(
        ...,
        description="OAuth2 client secret",
        json_schema_extra={"secret": True},
    )
    authorization_url: str = Field(..., description="OAuth2 authorization URL")
    token_url: str = Field(..., description="OAuth2 token URL")
    userinfo_url: str | None = Field(default=None, description="OAuth2 user info URL")
    scopes: list[str] = Field(default_factory=list, description="OAuth2 scopes")
    redirect_uri: str = Field(..., description="OAuth2 redirect URI")

    @field_validator("authorization_url", "token_url", "userinfo_url", "redirect_uri")
    @classmethod
    def validate_url(cls, v: str | None) -> str | None:
        """Validate URL format."""
        if v is None:
            return v
        if not v.startswith(("http://", "https://")):
            msg = "URL must start with http:// or https://"
            raise ValueError(msg)
        return v.rstrip("/")


class LDAPConfig(BaseConfig):
    """LDAP authentication configuration - consolidated pattern."""

    server_uri: str = Field(..., description="LDAP server URI")
    bind_dn: str = Field(..., description="LDAP bind DN")
    bind_password: str = Field(
        ...,
        description="LDAP bind password",
        json_schema_extra={"secret": True},
    )
    base_dn: str = Field(..., description="LDAP base DN for searches")
    user_filter: str = Field(
        default="(uid={username})",
        description="LDAP user search filter",
    )
    group_filter: str = Field(
        default="(member={user_dn})",
        description="LDAP group search filter",
    )

    # Connection options
    use_tls: bool = Field(default=True, description="Use TLS connection")
    tls_validate: bool = Field(default=True, description="Validate TLS certificates")
    timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Connection timeout in seconds",
    )

    @field_validator("server_uri")
    @classmethod
    def validate_ldap_uri(cls, v: str) -> str:
        """Validate LDAP URI format."""
        if not v.startswith(("ldap://", "ldaps://")):
            msg = "LDAP URI must start with ldap:// or ldaps://"
            raise ValueError(msg)
        return v.rstrip("/")


class BasicAuthConfig(BaseConfig):
    """Basic authentication configuration - consolidated pattern."""

    username: str = Field(..., description="Basic auth username")
    password: str = Field(
        ...,
        description="Basic auth password",
        json_schema_extra={"secret": True},
    )
    realm: str = Field(default="Restricted", description="Authentication realm")


class APIKeyConfig(BaseConfig):
    """API key authentication configuration - consolidated pattern."""

    api_key: str = Field(..., description="API key", json_schema_extra={"secret": True})
    header_name: str = Field(default="X-API-Key", description="Header name for API key")
    query_parameter: str = Field(
        default="api_key",
        description="Query parameter name for API key",
    )
    location: str = Field(default="header", description="API key location")

    @field_validator("location")
    @classmethod
    def validate_location(cls, v: str) -> str:
        """Validate API key location."""
        allowed_locations = {"header", "query", "cookie"}
        if v not in allowed_locations:
            msg = f"Location must be one of: {allowed_locations}"
            raise ValueError(msg)
        return v


class SessionConfig(BaseConfig):
    """Session configuration - consolidated pattern."""

    secret_key: str = Field(
        ...,
        description="Session secret key",
        json_schema_extra={"secret": True},
    )
    cookie_name: str = Field(default="session", description="Session cookie name")
    cookie_domain: str | None = Field(default=None, description="Session cookie domain")
    cookie_path: str = Field(default="/", description="Session cookie path")
    cookie_secure: bool = Field(default=True, description="Secure session cookies")
    cookie_httponly: bool = Field(default=True, description="HTTP-only session cookies")
    cookie_samesite: str = Field(default="lax", description="SameSite cookie attribute")
    max_age: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Session max age in seconds",
    )

    @field_validator("cookie_samesite")
    @classmethod
    def validate_samesite(cls, v: str) -> str:
        """Validate SameSite attribute."""
        allowed_values = {"strict", "lax", "none"}
        if v.lower() not in allowed_values:
            msg = f"SameSite must be one of: {allowed_values}"
            raise ValueError(msg)
        return v.lower()


__all__ = [
    "APIKeyConfig",
    "BasicAuthConfig",
    "JWTConfig",
    "LDAPConfig",
    "OAuth2Config",
    "SessionConfig",
]
