"""Typed post-run integrity and saveability for pytest-testmon SQLite DBs."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Self, override

from flext_core import r
from flext_infra import m, u
from flext_infra.base import s

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraTestmonCacheState(m.Value):
    """Decision record after a testmon DB integrity pass."""

    seed_needed: Annotated[bool, m.Field(description="No usable DB was present.")]
    restored_accepted: Annotated[
        bool, m.Field(description="An existing DB passed integrity checks.")
    ]
    changed: Annotated[
        bool, m.Field(description="DB content changed relative to pre-run digest.")
    ]
    saveable: Annotated[
        bool, m.Field(description="DB may be uploaded as a cache generation.")
    ]
    reason: Annotated[str, m.Field(min_length=1, description="Decisive state reason.")]


class FlextInfraTestmonDbInspector(s[FlextInfraTestmonCacheState]):
    """Checkpoint WAL, verify integrity, and compute saveability."""

    db_path: Annotated[Path, m.Field(description="Absolute .testmondata path.")]
    pre_run_digest: Annotated[
        str | None,
        m.Field(default=None, description="Hex digest of DB bytes before pytest."),
    ] = None
    run_succeeded: Annotated[
        bool, m.Field(description="Pytest finished green without timeout/cancel.")
    ]
    mode: Annotated[
        Literal["test"],
        m.Field(description="Runner mode; testmon-backed pytest runs only."),
    ]

    @u.model_validator(mode="after")
    def _validate_absolute_db(self) -> Self:
        """Reject relative or empty DB paths."""
        if not self.db_path.is_absolute():
            msg = "testmon db_path must be absolute"
            raise ValueError(msg)
        return self

    @staticmethod
    def digest_file(path: Path) -> str | None:
        """Return a stable hex digest for an existing regular file, else None."""
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            return None
        return u.Cli.sha256_file(path)

    def _reject(self, reason: str, *, seed_needed: bool) -> FlextInfraTestmonCacheState:
        """Build a non-saveable state with a decisive reason."""
        return FlextInfraTestmonCacheState(
            seed_needed=seed_needed,
            restored_accepted=False,
            changed=False,
            saveable=False,
            reason=reason,
        )

    def _validate_open_db(
        self, connection: sqlite3.Connection
    ) -> p.Result[FlextInfraTestmonCacheState] | None:
        """Return a reject Result when the open DB fails validation."""
        checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint is None or int(checkpoint[0]) != 0:
            return r[FlextInfraTestmonCacheState].ok(
                self._reject(
                    f"testmon wal_checkpoint busy={checkpoint!r}", seed_needed=False
                )
            )
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            return r[FlextInfraTestmonCacheState].ok(
                self._reject(f"testmon integrity_check={integrity!r}", seed_needed=True)
            )
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if not tables:
            return r[FlextInfraTestmonCacheState].ok(
                self._reject("testmon schema empty", seed_needed=True)
            )
        return None

    def _inspect_existing(self) -> p.Result[FlextInfraTestmonCacheState]:
        """Validate one on-disk DB after pytest has closed it."""
        path = self.db_path
        if path.is_symlink():
            return r[FlextInfraTestmonCacheState].ok(
                self._reject("testmon db is a symlink", seed_needed=True)
            )
        if not path.is_file() or path.stat().st_size == 0:
            return r[FlextInfraTestmonCacheState].ok(
                self._reject("testmon db missing or empty", seed_needed=True)
            )
        try:
            connection = sqlite3.connect(f"file:{path}?mode=rw", uri=True)
        except sqlite3.Error as exc:
            return r[FlextInfraTestmonCacheState].ok(
                self._reject(f"testmon db open failed: {exc}", seed_needed=True)
            )
        try:
            rejected = self._validate_open_db(connection)
        except sqlite3.Error as exc:
            return r[FlextInfraTestmonCacheState].ok(
                self._reject(f"testmon pragma failed: {exc}", seed_needed=True)
            )
        finally:
            connection.close()
        if rejected is not None:
            return rejected
        post_digest = self.digest_file(path)
        changed = post_digest is not None and post_digest != self.pre_run_digest
        seed_needed = self.pre_run_digest is None
        saveable = self.run_succeeded and (seed_needed or changed)
        if seed_needed and saveable:
            reason = "seed_ready"
        elif changed and saveable:
            reason = "changed_saveable"
        elif self.run_succeeded and not changed:
            reason = "unchanged"
        else:
            reason = "run_not_saveable"
        return r[FlextInfraTestmonCacheState].ok(
            FlextInfraTestmonCacheState(
                seed_needed=seed_needed,
                restored_accepted=self.pre_run_digest is not None,
                changed=changed,
                saveable=saveable,
                reason=reason,
            )
        )

    @override
    def execute(self) -> p.Result[FlextInfraTestmonCacheState]:
        """Return typed cache state after a successful testmon-backed pytest run."""
        if not self.run_succeeded:
            return r[FlextInfraTestmonCacheState].ok(
                self._reject("pytest run not successful", seed_needed=False)
            )
        return self._inspect_existing()


__all__: list[str] = ["FlextInfraTestmonCacheState", "FlextInfraTestmonDbInspector"]
