"""Render the canonical focused pytest cProfile artifact."""

from __future__ import annotations

import io
import os
import pstats
from pathlib import Path

from flext_infra import config


def main() -> int:
    """Render the latest focused pytest profile with config-owned policy."""
    report_root = Path.cwd().resolve() / ".reports" / "cprofile"
    profile_path = report_root / "pytest.pstats"
    output_path = report_root / "pytest.txt"
    if not profile_path.is_file():
        msg = f"cProfile artifact does not exist: {profile_path}"
        raise FileNotFoundError(msg)
    policy = config.Infra.tooling.tools.pytest
    stream = io.StringIO()
    stats = pstats.Stats(str(profile_path), stream=stream)
    stats.strip_dirs().sort_stats(policy.profile_sort).print_stats(policy.profile_limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.write_text(stream.getvalue(), encoding="utf-8")
    temporary.replace(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
