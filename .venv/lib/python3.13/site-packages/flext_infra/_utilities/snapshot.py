"""rsync-backed workspace snapshot helper.

Mirrors a workspace tree (typically ``flext`` → ``flext-base``) via
``rsync -a --delete`` so refactors can be tested in isolation before
propagating to the live workspace.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import shutil
from pathlib import Path

from flext_cli import cli, e, r
from flext_infra.protocols import p


class FlextInfraUtilitiesSnapshot:
    """rsync wrapper for sandbox workspace mirroring."""

    @staticmethod
    def rsync(*, src: Path, dst: Path) -> p.Result[Path]:
        """Mirror ``src`` to ``dst`` via ``rsync -a --delete``.

        Returns ``r.ok(dst)`` on success or a typed failure when rsync is
        not installed, the source is missing, or the rsync exits non-zero.
        Trailing slash on ``src`` is added so rsync copies the contents
        (not the directory itself) into ``dst``. Process execution flows
        through the canonical ``cli.run_raw`` helper.
        """
        rsync_bin = shutil.which("rsync")
        if rsync_bin is None:
            return e.fail_not_found("rsync executable", "PATH", result_type=r[Path])
        if not src.is_dir():
            return e.fail_not_found("snapshot source", str(src), result_type=r[Path])
        dst.mkdir(parents=True, exist_ok=True)
        outcome = cli.run_raw([
            rsync_bin,
            "-a",
            "--delete",
            f"{src.resolve()}/",
            str(dst.resolve()),
        ])
        if outcome.failure:
            return r[Path].fail_op("rsync snapshot", outcome.error or "")
        if outcome.value.exit_code != 0:
            return r[Path].fail_op("rsync snapshot", outcome.value.stderr.strip())
        return r[Path].ok(dst.resolve())


__all__: list[str] = ["FlextInfraUtilitiesSnapshot"]
