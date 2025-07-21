"""Oracle Integration Cloud (OIC) centralized patterns.

This module provides base classes and common patterns for Oracle Integration Cloud
integration across all FLEXT projects. It eliminates duplication of OIC
authentication, API client, and stream patterns.
"""

from __future__ import annotations

import base64
import logging
from abc import ABC
from abc import abstractmethod
from typing import Any

import requests
from pydantic import BaseModel
from pydantic import Field
from pydantic import SecretStr
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from flext_core.domain.types import ServiceResult

logger = logging.getLogger(__name__)


class OICAuthConfig(BaseModel):
    """Oracle Integration Cloud authentication configuration."""

    oauth_client_id: str = Field(..., description="IDCS OAuth2 client ID")
    oauth_client_secret: SecretStr = Field(..., description="IDCS OAuth2 client secret")
    oauth_token_url: str = Field(..., description="IDCS OAuth2 token endpoint")
    oauth_client_aud: str | None = Field(None, description="OAuth2 audience")
    oauth_scope: str = Field("", description="OAuth2 scope")

    class Config:
        extra = "forbid"


class OICConnectionConfig(BaseModel):
    """Oracle Integration Cloud connection configuration."""

    base_url: str = Field(..., description="Oracle OIC instance base URL")
    api_version: str = Field("v1", description="OIC API version")
    request_timeout: int = Field(30, ge=1, description="Request timeout in seconds")
    max_retries: int = Field(3, ge=0, description="Maximum retry attempts")

    def validate_base_url(self) -> ServiceResult[None]:
        """Validate OIC base URL format."""
        if not self.base_url.startswith(("http://", "https://")):
            return ServiceResult.fail("Base URL must start with http:// or https://")

        if not any(
            domain in self.base_url
            for domain in [
                "integration.ocp.oraclecloud.com",
                "integration.oci.oraclecloud.com",
                "integration.us.oraclecloud.com",
            ]
        ):
            logger.warning(
                "Base URL may not be a valid OIC endpoint: %s", self.base_url,
            )

        return ServiceResult.ok(None)

    class Config:
        extra = "forbid"


class BaseOICAuthenticator(ABC):
    """Base class for Oracle Integration Cloud authentication patterns."""

    def __init__(self, auth_config: OICAuthConfig) -> None:
        """Initialize OIC authenticator.

        Args:
            auth_config: OIC authentication configuration

        """
        self.auth_config = auth_config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._access_token: str | None = None

    @abstractmethod
    def get_oauth_scopes(self) -> str:
        """Get OAuth2 scopes for authentication.

        Returns:
            OAuth2 scope string

        """

    def build_oauth_scopes(self, client_aud: str | None = None) -> str:
        """Build OAuth2 scopes for Oracle Integration Cloud.

        Args:
            client_aud: OAuth2 audience (auto-detected if not provided)

        Returns:
            OAuth2 scope string

        """
        audience = client_aud or self.auth_config.oauth_client_aud

        if audience:
            # Build scope like: "audience:443urn:opc:resource:consumer:all audience:443/ic/api/"
            resource_aud = f"{audience}:443urn:opc:resource:consumer:all"
            api_aud = f"{audience}:443/ic/api/"
            return f"{resource_aud} {api_aud}"
        # Fallback to simple scope
        return self.auth_config.oauth_scope or "urn:opc:resource:consumer:all"

    def get_oauth_request_body(self) -> dict[str, Any]:
        """Generate OAuth2 request body for client credentials flow.

        Returns:
            Dict containing grant_type and scope for OAuth2 token request

        """
        return {
            "grant_type": "client_credentials",
            "scope": self.get_oauth_scopes(),
        }

    def encode_client_credentials(self) -> str:
        """Encode client credentials for HTTP Basic authentication.

        Returns:
            Base64 encoded client credentials

        """
        client_id = self.auth_config.oauth_client_id
        client_secret = self.auth_config.oauth_client_secret.get_secret_value()

        credentials = f"{client_id}:{client_secret}"
        return base64.b64encode(credentials.encode()).decode()

    def get_access_token(self) -> ServiceResult[str]:
        """Get access token using OAuth2 client credentials flow.

        Returns:
            ServiceResult containing access token or error

        """
        try:
            if not self.auth_config.oauth_token_url:
                return ServiceResult.fail("OAuth token URL not configured")

            # Encode client credentials for HTTP Basic authentication
            encoded_credentials = self.encode_client_credentials()

            headers = {
                "Authorization": f"Basic {encoded_credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            }

            # Make token request
            response = requests.post(
                self.auth_config.oauth_token_url,
                headers=headers,
                data=self.get_oauth_request_body(),
                timeout=30,
            )
            response.raise_for_status()

            token_data = response.json()
            self._access_token = token_data["access_token"]

            self.logger.info("OIC OAuth2 authentication successful")
            return ServiceResult.ok(self._access_token)

        except requests.exceptions.RequestException as e:
            error_msg = f"OIC OAuth2 authentication failed: {e}"
            self.logger.exception(error_msg)
            return ServiceResult.fail(error_msg)
        except KeyError as e:
            error_msg = f"Invalid OAuth2 response format: missing {e}"
            self.logger.exception(error_msg)
            return ServiceResult.fail(error_msg)
        except Exception as e:
            error_msg = f"OIC authentication error: {e}"
            self.logger.exception(error_msg)
            return ServiceResult.fail(error_msg)


class OICTapAuthenticator(BaseOICAuthenticator):
    """OIC Tap-specific authenticator with standard scopes."""

    def get_oauth_scopes(self) -> str:
        """Get OAuth2 scopes for OIC Tap."""
        return self.build_oauth_scopes()


class OICTargetAuthenticator(BaseOICAuthenticator):
    """OIC Target-specific authenticator with write permissions."""

    def get_oauth_scopes(self) -> str:
        """Get OAuth2 scopes for OIC Target."""
        return self.build_oauth_scopes()


class BaseOICClient(ABC):
    """Base class for Oracle Integration Cloud API clients."""

    @abstractmethod
    def get_base_url(self) -> str:
        """Get OIC API base URL."""
        ...

    def __init__(
        self,
        connection_config: OICConnectionConfig,
        authenticator: BaseOICAuthenticator,
    ) -> None:
        """Initialize OIC client.

        Args:
            connection_config: OIC connection configuration
            authenticator: OIC authenticator instance

        """
        self.connection_config = connection_config
        self.authenticator = authenticator
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._session: requests.Session | None = None

    def get_authenticated_session(self) -> ServiceResult[requests.Session]:
        """Get authenticated HTTP session.

        Returns:
            ServiceResult containing authenticated session or error

        """
        try:
            if not self._session:
                # Get OAuth2 token
                token_result = self.authenticator.get_access_token()
                if not token_result.is_success:
                    return ServiceResult.fail(
                        f"Authentication failed: {token_result.error}",
                    )

                # Create authenticated session
                self._session = requests.Session()
                self._session.headers.update(
                    {
                        "Authorization": f"Bearer {token_result.data}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                )

                # Configure session with default timeout - will be used in requests
                # Note: timeout is set per request, not on session

                # Setup retry strategy
                retry_strategy = Retry(
                    total=self.connection_config.max_retries,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["HEAD", "GET", "OPTIONS"],
                    backoff_factor=1,
                )

                adapter = HTTPAdapter(max_retries=retry_strategy)
                self._session.mount("http://", adapter)
                self._session.mount("https://", adapter)

            return ServiceResult.ok(self._session)

        except Exception as e:
            error_msg = f"Failed to create authenticated session: {e}"
            self.logger.exception(error_msg)
            return ServiceResult.fail(error_msg)

    def make_request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> ServiceResult[dict[str, Any]]:
        """Make authenticated request to OIC API.

        Args:
            method: HTTP method
            endpoint: API endpoint (relative to base URL)
            params: Query parameters
            data: Form data
            json: JSON data

        Returns:
            ServiceResult containing response data or error

        """
        try:
            session_result = self.get_authenticated_session()
            if not session_result.is_success:
                return ServiceResult.fail(
                    session_result.error or "Authentication failed",
                )

            session = session_result.data
            if session is None:
                return ServiceResult.fail("Failed to get authenticated session")

            # Build full URL
            base_url = self.get_base_url()
            url = (
                f"{base_url}{endpoint}"
                if endpoint.startswith("/")
                else f"{base_url}/{endpoint}"
            )

            # Make request
            response = session.request(
                method=method,
                url=url,
                params=params,
                data=data,
                json=json,
            )
            response.raise_for_status()

            # Parse response
            if response.headers.get("content-type", "").startswith("application/json"):
                return ServiceResult.ok(response.json())
            return ServiceResult.ok({"raw_content": response.text})

        except requests.exceptions.HTTPError as e:
            error_msg = f"OIC API HTTP error: {e}"
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_data = e.response.json()
                    error_msg = f"OIC API error: {error_data}"
                except (ValueError, KeyError):
                    error_msg = f"OIC API error: {e.response.text}"
            self.logger.exception(error_msg)
            return ServiceResult.fail(error_msg)
        except requests.exceptions.RequestException as e:
            error_msg = f"OIC API request failed: {e}"
            self.logger.exception(error_msg)
            return ServiceResult.fail(error_msg)
        except Exception as e:
            error_msg = f"OIC API client error: {e}"
            self.logger.exception(error_msg)
            return ServiceResult.fail(error_msg)

    def paginate_request(
        self,
        endpoint: str,
        page_size: int = 100,
        params: dict[str, Any] | None = None,
    ) -> ServiceResult[list[dict[str, Any]]]:
        """Paginate through OIC API responses.

        Args:
            endpoint: API endpoint to paginate
            page_size: Number of records per page
            params: Additional query parameters

        Returns:
            ServiceResult containing all paginated records or error

        """
        try:
            all_records = []
            offset = 0
            base_params = params or {}

            while True:
                # Prepare pagination parameters
                request_params = base_params.copy()
                request_params.update(
                    {
                        "offset": offset,
                        "limit": page_size,
                    },
                )

                # Make request
                response_result = self.make_request(
                    "GET", endpoint, params=request_params,
                )
                if not response_result.is_success:
                    return ServiceResult.fail(response_result.error or "Request failed")

                response_data = response_result.data
                if response_data is None:
                    return ServiceResult.fail("No response data received")

                items = response_data.get("items", [])

                # Add items to collection
                all_records.extend(items)

                # Check if we have more pages
                has_more = response_data.get("hasMore", False)
                if not has_more or len(items) < page_size:
                    break

                offset += page_size

            return ServiceResult.ok(all_records)

        except Exception as e:
            error_msg = f"OIC pagination failed: {e}"
            self.logger.exception(error_msg)
            return ServiceResult.fail(error_msg)


class OICTapClient(BaseOICClient):
    """OIC Tap client for data extraction."""

    def get_integrations(
        self,
        status_filter: list[str] | None = None,
        page_size: int = 100,
    ) -> ServiceResult[list[dict[str, Any]]]:
        """Get integration flows from OIC.

        Args:
            status_filter: Filter integrations by status
            page_size: Number of records per page

        Returns:
            ServiceResult containing integrations or error

        """
        params = {}

        if status_filter:
            params["q"] = f"status in ({','.join(status_filter)})"

        return self.paginate_request(
            "/integrations", page_size=page_size, params=params,
        )

    def get_connections(
        self,
        type_filter: list[str] | None = None,
        page_size: int = 100,
    ) -> ServiceResult[list[dict[str, Any]]]:
        """Get adapter connections from OIC.

        Args:
            type_filter: Filter connections by adapter type
            page_size: Number of records per page

        Returns:
            ServiceResult containing connections or error

        """
        params = {}

        if type_filter:
            params["q"] = f"adapterType in ({','.join(type_filter)})"

        return self.paginate_request("/connections", page_size=page_size, params=params)

    def get_packages(self, page_size: int = 100) -> ServiceResult[list[dict[str, Any]]]:
        """Get integration packages from OIC.

        Args:
            page_size: Number of records per page

        Returns:
            ServiceResult containing packages or error

        """
        return self.paginate_request("/packages", page_size=page_size)

    def get_lookups(self, page_size: int = 100) -> ServiceResult[list[dict[str, Any]]]:
        """Get lookup tables from OIC.

        Args:
            page_size: Number of records per page

        Returns:
            ServiceResult containing lookups or error

        """
        return self.paginate_request("/lookups", page_size=page_size)


class OICTargetClient(BaseOICClient):
    """OIC Target client for data loading."""

    def create_integration(
        self, integration_data: dict[str, Any],
    ) -> ServiceResult[dict[str, Any]]:
        """Create integration in OIC.

        Args:
            integration_data: Integration configuration data

        Returns:
            ServiceResult containing created integration or error

        """
        return self.make_request("POST", "/integrations", json=integration_data)

    def update_integration(
        self,
        integration_id: str,
        integration_data: dict[str, Any],
    ) -> ServiceResult[dict[str, Any]]:
        """Update integration in OIC.

        Args:
            integration_id: Integration ID to update
            integration_data: Updated integration data

        Returns:
            ServiceResult containing updated integration or error

        """
        endpoint = f"/integrations/{integration_id}"
        return self.make_request("PUT", endpoint, json=integration_data)

    def create_connection(
        self, connection_data: dict[str, Any],
    ) -> ServiceResult[dict[str, Any]]:
        """Create connection in OIC.

        Args:
            connection_data: Connection configuration data

        Returns:
            ServiceResult containing created connection or error

        """
        return self.make_request("POST", "/connections", json=connection_data)
