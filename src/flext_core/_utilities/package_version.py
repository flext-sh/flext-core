"""Package version derivation from installed distribution metadata.

The root ``__version__`` module of every FLEXT package binds its dunders from
this owner at package import, so it reads only the standard library and
``packaging``: it never imports a facade.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from email.utils import parseaddr
from importlib.metadata import PackageMetadata

from packaging.version import Version


class FlextUtilitiesPackageVersion:
    """Package identity derived from one distribution's metadata."""

    @staticmethod
    def package_author(package_metadata: PackageMetadata) -> tuple[str, str]:
        """Return the first normalized author identity from package metadata.

        Returns:
            The author name and email address.

        Raises:
            ValueError: If the Author-Email metadata carries no address.

        """
        raw_email = package_metadata.get("Author-Email", "")
        author_name, author_email = parseaddr(raw_email, strict=True)
        if raw_email and not author_email:
            msg = f"invalid Author-Email package metadata: {raw_email!r}"
            raise ValueError(msg)
        return (
            package_metadata.get("Author", "") or author_name or author_email,
            author_email,
        )

    @staticmethod
    def package_homepage(package_metadata: PackageMetadata) -> str:
        """Return the legacy Home-Page or labeled Homepage project URL.

        Returns:
            The homepage URL, empty when the metadata declares none.

        Raises:
            ValueError: If a Homepage project URL entry carries no URL.

        """
        if homepage := package_metadata.get("Home-Page", ""):
            return homepage
        for project_url in package_metadata.get_all("Project-URL") or ():
            project_url_text = str(project_url)
            label, separator, url = project_url_text.partition(",")
            if label.strip().casefold() != "homepage":
                continue
            if not separator or not url.strip():
                msg = f"invalid Homepage project URL metadata: {project_url_text!r}"
                raise ValueError(msg)
            return url.strip()
        return ""

    @staticmethod
    def package_version_info(version: str) -> tuple[int, int, int]:
        """Return the exact three-component PEP 440 release tuple.

        Returns:
            The integer major, minor and patch release components.

        Raises:
            ValueError: If the version is not PEP 440 or its release does not
                have exactly three components.

        """
        try:
            major, minor, patch = Version(version).release
        except ValueError as exc:
            msg = f"invalid three-part semantic version metadata: {version!r}"
            raise ValueError(msg) from exc
        return major, minor, patch

    class PackageVersion:
        """MRO base deriving package metadata for each class declaring ``_metadata``.

        A subclass declares only ``_metadata``; every public value is derived
        once when that subclass is created, so invalid metadata fails at its
        definition.
        """

        _metadata: PackageMetadata
        __version__: str = ""
        __version_info__: tuple[int, int, int]
        __title__: str = ""
        __description__: str = ""
        __author__: str = ""
        __author_email__: str = ""
        __license__: str = ""
        __url__: str = ""

        def __init_subclass__(cls, **kwargs: str | float | bool | None) -> None:
            """Derive the public metadata when a subclass declares ``_metadata``."""
            super().__init_subclass__(**kwargs)
            if "_metadata" not in cls.__dict__:
                return
            owner = FlextUtilitiesPackageVersion
            package_metadata = cls._metadata
            cls.__version__ = package_metadata["Version"]
            cls.__version_info__ = owner.package_version_info(cls.__version__)
            cls.__title__ = package_metadata.get("Name", "")
            cls.__description__ = package_metadata.get("Summary", "")
            cls.__author__, cls.__author_email__ = owner.package_author(
                package_metadata,
            )
            cls.__license__ = package_metadata.get(
                "License-Expression",
                "",
            ) or package_metadata.get("License", "")
            cls.__url__ = owner.package_homepage(package_metadata)


__all__: list[str] = ["FlextUtilitiesPackageVersion"]
