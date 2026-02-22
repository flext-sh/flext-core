from __future__ import annotations

from pathlib import Path

from scripts.dependencies import sync_dep_paths as legacy


class DepPathSyncer:
    def rewrite(
        self,
        pyproject_path: Path,
        *,
        mode: str,
        is_root: bool = False,
        dry_run: bool = False,
    ) -> list[str]:
        return legacy.rewrite_dep_paths(
            pyproject_path,
            mode=mode,
            is_root=is_root,
            dry_run=dry_run,
        )

    def run(self) -> int:
        return legacy._main()


def extract_dep_name(raw_path: str) -> str:
    return legacy.extract_dep_name(raw_path)


def main() -> int:
    return DepPathSyncer().run()


if __name__ == "__main__":
    raise SystemExit(main())
