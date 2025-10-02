"""Unified version metadata loader for flext-core and dependants."""

from __future__ import annotations

import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Self

from flext_core.exceptions import FlextExceptions

__all__ = [
    "STANDARD_VERSION_EXPORTS",
    "VERSION",
    "FlextProjectMetadata",
    "FlextProjectPerson",
    "FlextProjectVersion",
    "FlextVersion",
    "__version__",
    "__version_info__",
    "build_metadata_exports",
    "get_project_license",
    "get_project_metadata",
    "get_project_name",
    "get_project_version",
    "load_project_metadata",
]


@dataclass(frozen=True, slots=True)
class FlextProjectPerson:
    """Representation of an individual declared in project metadata."""

    name: str
    email: str | None = None


@dataclass(frozen=True, slots=True)
class FlextProjectMetadata:
    """Normalized metadata extracted from ``pyproject.toml``."""

    project_root: Path
    name: str
    version: str
    description: str | None
    license: str | None
    requires_python: str | None
    authors: tuple[FlextProjectPerson, ...]
    maintainers: tuple[FlextProjectPerson, ...]
    keywords: tuple[str, ...]
    classifiers: tuple[str, ...]
    urls: Mapping[str, str]
    readme: str | None

    @property
    def version_tuple(self) -> tuple[int | str, ...]:
        parts = [
            int(part) if part.isdigit() else part for part in self.version.split(".")
        ]
        return tuple(parts)

    @property
    def primary_author(self) -> FlextProjectPerson | None:
        return self.authors[0] if self.authors else None

    @property
    def primary_maintainer(self) -> FlextProjectPerson | None:
        return self.maintainers[0] if self.maintainers else None

    def require_description(self) -> str:
        if not self.description:
            msg = "pyproject.toml must declare a project description"
            raise FlextExceptions.ConfigurationError(
                msg,
                config_key="description",
                config_file="pyproject.toml",
            )
        return self.description

    def require_license(self) -> str:
        if not self.license:
            msg = "pyproject.toml must declare a license text"
            raise FlextExceptions.ConfigurationError(
                msg,
                config_key="license",
                config_file="pyproject.toml",
            )
        return self.license

    def require_primary_author(self) -> FlextProjectPerson:
        author = self.primary_author
        if author is None:
            msg = "pyproject.toml must declare at least one author"
            raise FlextExceptions.ConfigurationError(
                msg,
                config_key="authors",
                config_file="pyproject.toml",
            )
        return author

    def require_primary_maintainer(self) -> FlextProjectPerson:
        maintainer = self.primary_maintainer
        if maintainer is None:
            msg = "pyproject.toml must declare at least one maintainer"
            raise FlextExceptions.ConfigurationError(
                msg,
                config_key="maintainers",
                config_file="pyproject.toml",
            )
        return maintainer


STANDARD_VERSION_EXPORTS: tuple[str, ...] = ("__version__", "__version_info__")


@dataclass(frozen=True, slots=True)
class FlextProjectVersion:
    """Structured view of project metadata sourced from ``pyproject.toml``."""

    metadata: FlextProjectMetadata

    @property
    def project(self) -> str:
        return self.metadata.name

    @property
    def version(self) -> str:
        return self.metadata.version

    @property
    def version_info(self) -> tuple[int | str, ...]:
        return self.metadata.version_tuple

    @property
    def description(self) -> str | None:
        return self.metadata.description

    @property
    def license(self) -> str | None:
        return self.metadata.license

    @property
    def requires_python(self) -> str | None:
        return self.metadata.requires_python

    @property
    def author(self) -> str:
        return self.metadata.require_primary_author().name

    @property
    def author_email(self) -> str | None:
        return self.metadata.require_primary_author().email

    @property
    def maintainer(self) -> str:
        return self.metadata.require_primary_maintainer().name

    @property
    def maintainer_email(self) -> str | None:
        return self.metadata.require_primary_maintainer().email

    @property
    def author_names(self) -> tuple[str, ...]:
        return tuple(person.name for person in self.metadata.authors)

    @property
    def maintainer_names(self) -> tuple[str, ...]:
        return tuple(person.name for person in self.metadata.maintainers)

    @property
    def urls(self) -> Mapping[str, str]:
        return self.metadata.urls

    def as_dict(self) -> dict[str, object]:
        return {
            "project": self.project,
            "version": self.version,
            "version_info": self.version_info,
            "description": self.description,
            "license": self.license,
            "requires_python": self.requires_python,
            "author": self.author,
            "author_email": self.author_email,
            "maintainer": self.maintainer,
            "maintainer_email": self.maintainer_email,
            "urls": dict(self.urls),
        }

    def export_globals(
        self,
        *,
        include: Iterable[str] | None = None,
    ) -> dict[str, object]:
        names = tuple(include) if include else STANDARD_VERSION_EXPORTS
        exports: dict[str, object] = {}
        for name in names:
            match name:
                case "__version__":
                    exports[name] = self.version
                case "__version_info__" | "__version_tuple__":
                    exports[name] = self.version_info
                case _:
                    message = f"Unsupported export name '{name}'"
                    raise FlextExceptions.ConfigurationError(
                        message,
                        config_key=name,
                    )
        return exports

    @classmethod
    def from_metadata(cls, metadata: FlextProjectMetadata) -> Self:
        return cls(metadata=metadata)

    @classmethod
    def from_file(cls, package_file: str | Path) -> Self:
        metadata = load_project_metadata(package_file)
        return cls.from_metadata(metadata)

    @classmethod
    def current(cls) -> Self:
        return cls.from_file(__file__)


class FlextVersion(FlextProjectVersion):
    """Generic metadata helper backed by ``pyproject.toml``."""

    __slots__ = ("_exports", "metadata")

    _STANDARD_EXPORTS: ClassVar[tuple[str, ...]] = STANDARD_VERSION_EXPORTS

    def __init__(
        self, metadata: FlextProjectMetadata, exports: Mapping[str, object]
    ) -> None:
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "_exports", MappingProxyType(dict(exports)))

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------
    @classmethod
    def current(cls) -> Self:
        """Instantiate using the project metadata for ``flext_core`` itself."""
        return cls.from_file(__file__)

    @classmethod
    def from_file(cls, package_file: str | Path) -> Self:
        metadata = cls._load_project_metadata(Path(package_file))
        exports = cls._build_exports(metadata)
        return cls(metadata=metadata, exports=exports)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def project(self) -> str:
        return self.metadata.name

    @property
    def version(self) -> str:
        return self.metadata.version

    @property
    def version_info(self) -> tuple[int | str, ...]:
        return self.metadata.version_tuple

    def export(self, name: str, default: Any | None = None) -> Any | None:
        return self._exports.get(name, default)

    def exports(
        self,
        include: Iterable[str] | None = None,
        *,
        full: bool = False,
    ) -> dict[str, object]:
        if full:
            return dict(self._exports)
        names = tuple(include) if include else self._STANDARD_EXPORTS
        result: dict[str, object] = {}
        for key in names:
            match key:
                case "__version__":
                    result[key] = self.version
                case "__version_info__" | "__version_tuple__":
                    result[key] = self.version_info
                case _:
                    result[key] = self.export(key)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_project_metadata(package_file: Path) -> FlextProjectMetadata:
        project_root = FlextVersion._find_project_root(package_file.resolve())
        return FlextVersion._load_metadata_from_root(project_root)

    @staticmethod
    @cache
    def _load_metadata_from_root(project_root: Path) -> FlextProjectMetadata:
        pyproject_path = project_root / "pyproject.toml"
        if not pyproject_path.exists():
            msg = f"pyproject.toml not found for project starting at {project_root}"
            raise FlextExceptions.NotFoundError(
                msg,
                resource_type="file",
                resource_id=str(pyproject_path),
            )

        with pyproject_path.open("rb") as file_pointer:
            data = tomllib.load(file_pointer)

        project_table = FlextVersion._extract_project_table(data)

        name = FlextVersion._require_string(project_table, "name")
        version = FlextVersion._require_string(project_table, "version")
        description = FlextVersion._optional_string(project_table.get("description"))
        requires_python = FlextVersion._optional_string(
            project_table.get("requires-python")
        )
        readme = FlextVersion._optional_string(project_table.get("readme"))
        license_value = FlextVersion._parse_license(project_table.get("license"))

        authors = FlextVersion._parse_people(project_table.get("authors"))
        maintainers = FlextVersion._parse_people(project_table.get("maintainers"))
        keywords = FlextVersion._parse_string_iterable(project_table.get("keywords"))
        classifiers = FlextVersion._parse_string_iterable(
            project_table.get("classifiers")
        )

        urls_raw = project_table.get("urls", {})
        if isinstance(urls_raw, Mapping):
            urls = MappingProxyType({
                str(key): str(value) for key, value in urls_raw.items()
            })
        else:
            urls = MappingProxyType({})

        return FlextProjectMetadata(
            project_root=project_root,
            name=name,
            version=version,
            description=description,
            license=license_value,
            requires_python=requires_python,
            authors=authors,
            maintainers=maintainers,
            keywords=keywords,
            classifiers=classifiers,
            urls=urls,
            readme=readme,
        )

    @staticmethod
    def _build_exports(metadata: FlextProjectMetadata) -> dict[str, object]:
        author = metadata.require_primary_author()
        maintainer = metadata.require_primary_maintainer()
        license_text = metadata.require_license()
        description = metadata.require_description()

        return {
            "__flext_metadata__": metadata,
            "__project__": metadata.name,
            "__version__": metadata.version,
            "__version_info__": metadata.version_tuple,
            "__version_tuple__": metadata.version_tuple,
            "__description__": description,
            "__license__": license_text,
            "__author__": author.name,
            "__author_email__": author.email or "",
            "__authors__": tuple(person.name for person in metadata.authors),
            "__maintainer__": maintainer.name,
            "__maintainer_email__": maintainer.email or author.email or "",
            "__maintainers__": tuple(person.name for person in metadata.maintainers),
            "__requires_python__": metadata.requires_python or "",
            "__urls__": dict(metadata.urls),
        }

    @staticmethod
    def _extract_project_table(data: Mapping[str, object]) -> Mapping[str, object]:
        project_table = data.get("project")
        if isinstance(project_table, Mapping):
            return project_table

        tool_table = data.get("tool")
        if isinstance(tool_table, Mapping):
            poetry_table = tool_table.get("poetry")
            if isinstance(poetry_table, Mapping):
                return poetry_table

        msg = "pyproject.toml must define a [project] or [tool.poetry] table"
        raise FlextExceptions.ConfigurationError(
            msg,
            config_key="project",
            config_file="pyproject.toml",
        )

    @staticmethod
    def _find_project_root(path: Path) -> Path:
        for candidate in (path, *path.parents):
            if (candidate / "pyproject.toml").exists():
                return candidate
        msg = f"Unable to locate project root for {path}"
        raise FlextExceptions.NotFoundError(
            msg,
            resource_type="directory",
            resource_id=str(path),
        )

    @staticmethod
    def _require_string(data: Mapping[str, object], key: str) -> str:
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            msg = f"Required string field '{key}' missing in pyproject.toml"
            raise FlextExceptions.ConfigurationError(
                msg,
                config_key=key,
                config_file="pyproject.toml",
            )
        return value.strip()

    @staticmethod
    def _optional_string(value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            trimmed = value.strip()
            return trimmed or None
        return None

    @staticmethod
    def _parse_license(raw_license: object) -> str | None:
        if raw_license is None:
            return None
        if isinstance(raw_license, str):
            return raw_license.strip() or None
        if isinstance(raw_license, Mapping):
            text = raw_license.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
            file_reference = raw_license.get("file")
            if isinstance(file_reference, str) and file_reference.strip():
                return file_reference.strip()
        msg = "License information must contain either text or file entry"
        raise FlextExceptions.ConfigurationError(
            msg,
            config_key="license",
            config_file="pyproject.toml",
        )

    @staticmethod
    def _parse_people(raw_people: object) -> tuple[FlextProjectPerson, ...]:
        if raw_people is None:
            return ()
        if not isinstance(raw_people, Iterable):
            msg = "Expected a list of authors/maintainers in pyproject.toml"
            raise FlextExceptions.TypeError(
                msg,
                expected_type="Iterable",
                actual_type=type(raw_people).__name__,
            )

        people: list[FlextProjectPerson] = []
        for entry in raw_people:
            if isinstance(entry, Mapping):
                name = FlextVersion._require_string(entry, "name")
                email_raw = entry.get("email")
                email = (
                    email_raw.strip()
                    if isinstance(email_raw, str) and email_raw.strip()
                    else None
                )
                people.append(FlextProjectPerson(name=name, email=email))
            elif isinstance(entry, str):
                name, email = FlextVersion._split_name_email(entry)
                people.append(FlextProjectPerson(name=name, email=email))
            else:
                msg = "Author and maintainer entries must be strings or tables with name/email"
                raise FlextExceptions.TypeError(
                    msg,
                    expected_type="str | dict",
                    actual_type=type(entry).__name__,
                )
        return tuple(people)

    @staticmethod
    def _split_name_email(value: str) -> tuple[str, str | None]:
        candidate = value.strip()
        if not candidate:
            msg = "Empty author/maintainer entry is not allowed"
            raise FlextExceptions.ValidationError(
                msg,
                field="author/maintainer",
                value=candidate,
            )
        if "<" in candidate and candidate.endswith(">"):
            name_part, email_part = candidate.split("<", 1)
            return name_part.strip(), email_part.rstrip(">").strip() or None
        return candidate, None

    @staticmethod
    def _parse_string_iterable(raw_values: object) -> tuple[str, ...]:
        if raw_values is None:
            return ()
        if isinstance(raw_values, Iterable) and not isinstance(
            raw_values, (str, bytes)
        ):
            parsed: list[str] = [
                item.strip()
                for item in raw_values
                if isinstance(item, str) and item.strip()
            ]
            return tuple(parsed)
        msg = "Expected an iterable of strings in pyproject metadata"
        actual_type = (
            type(raw_values).__name__ if raw_values is not None else "NoneType"
        )
        raise FlextExceptions.TypeError(
            msg,
            expected_type="Iterable[str]",
            actual_type=actual_type,
        )


VERSION: FlextVersion = FlextVersion.current()

__version__: str = VERSION.version
__version_info__: tuple[int | str, ...] = VERSION.version_info


def load_project_metadata(package_file: str | Path) -> FlextProjectMetadata:
    """Load metadata for the project containing ``package_file``."""
    return FlextVersion._load_project_metadata(Path(package_file))


def get_project_metadata(package_file: str | Path) -> FlextProjectMetadata:
    """Alias for :func:`load_project_metadata`."""
    return load_project_metadata(package_file)


def get_project_version(package_file: str | Path) -> str:
    """Return the project version string."""
    return load_project_metadata(package_file).version


def get_project_name(package_file: str | Path) -> str:
    """Return the project name."""
    return load_project_metadata(package_file).name


def get_project_license(package_file: str | Path) -> str | None:
    """Return the project license."""
    return load_project_metadata(package_file).license


def build_metadata_exports(package_file: str | Path) -> dict[str, object]:
    """Return module-level metadata exports for ``package_file``."""
    return FlextVersion.from_file(package_file).exports(full=True)
