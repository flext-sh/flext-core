"""SSL utilities for secure connections."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import grpc  # type: ignore[import-untyped]


def _create_ssl_credentials(
    root_certificates: bytes | None = None,
    private_key: bytes | None = None,
    certificate_chain: bytes | None = None,
) -> Any:
    """Create SSL credentials for gRPC channels.

    Args:
    ----
        root_certificates: PEM-encoded root certificates
        private_key: PEM-encoded private key for client auth
        certificate_chain: PEM-encoded certificate chain for client auth

    Returns:
    -------
        gRPC channel credentials for SSL/TLS connections

    """
    try:
        return grpc.ssl_channel_credentials(
            root_certificates=root_certificates,
            private_key=private_key,
            certificate_chain=certificate_chain,
        )
    except Exception:
        # Fallback to default SSL credentials
        return grpc.ssl_channel_credentials()


def load_ssl_credentials_from_files(
    ca_cert_path: str | None = None,
    client_cert_path: str | None = None,
    client_key_path: str | None = None,
) -> Any:
    """Load SSL credentials from certificate files.

    Args:
    ----
        ca_cert_path: Path to CA certificate file
        client_cert_path: Path to client certificate file
        client_key_path: Path to client private key file

    Returns:
    -------
        gRPC channel credentials loaded from files

    """
    root_certificates = None
    private_key = None
    certificate_chain = None

    if ca_cert_path and Path(ca_cert_path).exists():
        with Path(ca_cert_path).open("rb") as f:
            root_certificates = f.read()

    if client_key_path and Path(client_key_path).exists():
        with Path(client_key_path).open("rb") as f:
            private_key = f.read()

    if client_cert_path and Path(client_cert_path).exists():
        with Path(client_cert_path).open("rb") as f:
            certificate_chain = f.read()

    return _create_ssl_credentials(
        root_certificates=root_certificates,
        private_key=private_key,
        certificate_chain=certificate_chain,
    )


async def create_secure_grpc_channel_async(target: str) -> Any:
    """Create a secure gRPC channel asynchronously."""
    try:
        credentials = _create_ssl_credentials()
        return grpc.aio.secure_channel(target, credentials)
    except Exception:
        # Fallback to insecure channel for development
        return grpc.aio.insecure_channel(target)


def get_grpc_channel_target(host: str, port: int) -> str:
    """Get gRPC channel target string."""
    return f"{host}:{port}"
