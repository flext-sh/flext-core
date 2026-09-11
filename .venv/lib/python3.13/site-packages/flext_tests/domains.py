"""Test domain facade for FLEXT ecosystem tests."""

from __future__ import annotations

from pathlib import Path

from flext_cli import u as cli_u
from flext_tests import p, r, t


class FlextTestsDomains:
    """Test domain objects and fixtures."""

    @staticmethod
    def fixture_filename(group: str, kind: str, file_extension: str = ".ldif") -> str:
        """Build the canonical fixture filename for a group and kind."""
        return f"{group}_{kind}_fixtures{file_extension}"

    @classmethod
    def _resolve_fixture_path(
        cls,
        group: str,
        kind: str,
        *,
        fixtures_root: Path | None = None,
        file_extension: str = ".ldif",
    ) -> Path:
        """Compute the candidate fixture path without checking existence."""
        fixtures_root = fixtures_root or Path.cwd()
        return (
            fixtures_root
            / group
            / cls.fixture_filename(group, kind, file_extension=file_extension)
        )

    @classmethod
    def fixture_path(
        cls,
        group: str,
        kind: str,
        *,
        fixtures_root: Path | None = None,
        file_extension: str = ".ldif",
    ) -> Path:
        """Return the on-disk fixture path, raising if it is missing."""
        file_path = cls._resolve_fixture_path(
            group, kind, fixtures_root=fixtures_root, file_extension=file_extension
        )
        if not file_path.exists():
            msg = f"Fixture file not found: {file_path}"
            raise FileNotFoundError(msg)
        return file_path

    @classmethod
    def load_fixture(
        cls,
        group: str,
        kind: str,
        *,
        fixtures_root: Path | None = None,
        file_extension: str = ".ldif",
    ) -> str:
        """Read and return the text content of a fixture file."""
        text_result: p.Result[str] = cli_u.Cli.files_read_text(
            cls.fixture_path(
                group, kind, fixtures_root=fixtures_root, file_extension=file_extension
            )
        )
        if text_result.failure:
            raise ValueError(text_result.error or "Fixture file read failed")
        text: str = text_result.value
        return text

    @classmethod
    def fixture_exists(
        cls,
        group: str,
        kind: str,
        *,
        fixtures_root: Path | None = None,
        file_extension: str = ".ldif",
    ) -> bool:
        """Return whether the fixture file exists on disk."""
        return cls._resolve_fixture_path(
            group, kind, fixtures_root=fixtures_root, file_extension=file_extension
        ).exists()

    @classmethod
    def available_fixture_servers(
        cls, *, fixtures_root: Path | None = None
    ) -> t.StrSequence:
        """List fixture server group directories under the root."""
        fixtures_root = fixtures_root or Path.cwd()
        if not fixtures_root.exists():
            return ()
        return tuple(
            sorted(
                directory.name
                for directory in fixtures_root.iterdir()
                if directory.is_dir()
            )
        )

    @classmethod
    def available_fixture_types(
        cls,
        group: str,
        *,
        fixtures_root: Path | None = None,
        file_extension: str = ".ldif",
    ) -> t.StrSequence:
        """List available fixture kinds for one server group."""
        fixtures_root = fixtures_root or Path.cwd()
        server_dir = fixtures_root / group
        if not server_dir.exists():
            return ()
        prefix = f"{group}_"
        suffix = f"_fixtures{file_extension}"
        return tuple(
            sorted(
                name[len(prefix) : -len(suffix)]
                for name in sorted(entry.name for entry in server_dir.iterdir())
                if name.startswith(prefix) and name.endswith(suffix)
            )
        )

    @classmethod
    def load_server_fixtures(
        cls,
        group: str,
        *,
        fixtures_root: Path | None = None,
        file_extension: str = ".ldif",
    ) -> t.MappingKV[str, str]:
        """Load every fixture kind for one server group as a mapping."""
        return {
            fixture_type: cls.load_fixture(
                group,
                fixture_type,
                fixtures_root=fixtures_root,
                file_extension=file_extension,
            )
            for fixture_type in cls.available_fixture_types(
                group, fixtures_root=fixtures_root, file_extension=file_extension
            )
        }

    @staticmethod
    def default_handler_case_specs() -> t.SequenceOf[t.Tests.HandlerCaseSpec]:
        """Create shared handler test-case specs for service-base tests."""
        return [
            {
                "handler_id": "success_command",
                "handler_type": "COMMAND",
                "expected_result": "Handled: test",
                "description": "Command handler success",
            },
            {
                "handler_id": "success_query",
                "handler_type": "QUERY",
                "expected_result": "Handled: query",
                "description": "Query handler success",
            },
            {
                "handler_id": "success_event",
                "handler_type": "EVENT",
                "expected_result": "Handled: event",
                "description": "Event handler success",
            },
            {
                "handler_id": "fail_command",
                "handler_type": "COMMAND",
                "should_fail": True,
                "error_message": "Command failed",
                "description": "Command handler failure",
            },
            {
                "handler_id": "fail_query",
                "handler_type": "QUERY",
                "should_fail": True,
                "error_message": "Query failed",
                "description": "Query handler failure",
            },
        ]

    @staticmethod
    def create_result_ok[ValueT](value: ValueT) -> p.Result[ValueT]:
        """Create a generic successful result for test flows."""
        return r[ValueT].ok(value)

    @staticmethod
    def create_result_failure(
        message: str,
        *,
        error_code: str = "TEST_ERROR",
        error_data: t.JsonMapping | t.ConfigModelInput | None = None,
    ) -> p.Result[t.Tests.TestobjectSerializable]:
        """Create a generic failed result for test flows."""
        return r[t.Tests.TestobjectSerializable].fail(
            message, error_code=error_code, error_data=error_data
        )

    @staticmethod
    def valid_email_cases() -> t.SequenceOf[tuple[str, bool]]:
        """Get valid email test cases."""
        return [
            ("test@example.com", True),
            ("user.name@domain.co.uk", True),
            ("test+tag@example.com", True),
            ("invalid-email", False),
            ("@example.com", False),
            ("test@", False),
            ("", False),
        ]

    class BoundFixtures:
        """Bound fixture loader for one fixture root and extension."""

        def __init__(
            self, fixtures_root: Path, *, file_extension: str = ".ldif"
        ) -> None:
            """Store the fixtures root and default file extension."""
            self._fixtures_root: Path = fixtures_root
            self._file_extension: str = file_extension

        def fixture_path(self, group: str, kind: str) -> Path:
            """Return the on-disk fixture path, raising if it is missing."""
            return FlextTestsDomains.fixture_path(
                group,
                kind,
                fixtures_root=self._fixtures_root,
                file_extension=self._file_extension,
            )

        def load_fixture(self, group: str, kind: str) -> str:
            """Read and return the text content of a fixture file."""
            return FlextTestsDomains.load_fixture(
                group,
                kind,
                fixtures_root=self._fixtures_root,
                file_extension=self._file_extension,
            )

        def fixture_exists(self, group: str, kind: str) -> bool:
            """Return whether the fixture file exists on disk."""
            return FlextTestsDomains.fixture_exists(
                group,
                kind,
                fixtures_root=self._fixtures_root,
                file_extension=self._file_extension,
            )

        def available_fixture_servers(self) -> t.StrSequence:
            """List fixture server group directories under the root."""
            return FlextTestsDomains.available_fixture_servers(
                fixtures_root=self._fixtures_root
            )

        def available_fixture_types(self, group: str) -> t.StrSequence:
            """List available fixture kinds for one server group."""
            return FlextTestsDomains.available_fixture_types(
                group,
                fixtures_root=self._fixtures_root,
                file_extension=self._file_extension,
            )

        def load_server_fixtures(self, group: str) -> t.MappingKV[str, str]:
            """Load every fixture kind for one server group as a mapping."""
            return FlextTestsDomains.load_server_fixtures(
                group,
                fixtures_root=self._fixtures_root,
                file_extension=self._file_extension,
            )

        def load_fixture_kind(self, kind: str) -> t.MappingKV[str, str]:
            """Load every server's fixture for one kind as a mapping."""
            return {
                group: self.load_fixture(group, kind)
                for group in self.available_fixture_servers()
                if self.fixture_exists(group, kind)
            }

        def load_all(self) -> t.MappingKV[str, t.MappingKV[str, str]]:
            """Load every server and kind fixture as a nested mapping."""
            return {
                group: self.load_server_fixtures(group)
                for group in self.available_fixture_servers()
            }

        def pytest_params_for_group(self, group: str) -> t.StrPairSequence:
            """Return pytest params for one server group's fixtures."""
            return [
                (fixture_type, self.load_fixture(group, fixture_type))
                for fixture_type in self.available_fixture_types(group)
            ]

        def all_pytest_params(self) -> t.SequenceOf[tuple[str, str, str]]:
            """Return pytest params for every server group and kind."""
            return [
                (group, fixture_type, self.load_fixture(group, fixture_type))
                for group in self.available_fixture_servers()
                for fixture_type in self.available_fixture_types(group)
            ]

    @classmethod
    def bind(
        cls, fixtures_root: Path, *, file_extension: str = ".ldif"
    ) -> BoundFixtures:
        """Return a BoundFixtures view anchored at the given root."""
        return cls.BoundFixtures(fixtures_root, file_extension=file_extension)


td = FlextTestsDomains


__all__: list[str] = ["FlextTestsDomains", "td"]
