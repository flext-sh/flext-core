from __future__ import annotations

from typing import Any

from scripts.dependencies import sync_extra_paths_from_deps as legacy


class ExtraPathsSyncer:
    def run(self) -> int:
        return legacy.main()


def get_dep_paths(doc: Any, *, is_root: bool = False) -> list[str]:
    return legacy.get_dep_paths(doc, is_root=is_root)


PYRIGHT_BASE_PROJECT = legacy.PYRIGHT_BASE_PROJECT
MYPY_BASE_PROJECT = legacy.MYPY_BASE_PROJECT


def main() -> int:
    return ExtraPathsSyncer().run()


if __name__ == "__main__":
    raise SystemExit(main())
