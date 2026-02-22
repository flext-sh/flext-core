from __future__ import annotations

from scripts.dependencies import sync_internal_deps as legacy


class InternalDepsSyncer:
    def run(self) -> int:
        return legacy._main()


def main() -> int:
    return InternalDepsSyncer().run()


if __name__ == "__main__":
    raise SystemExit(main())
