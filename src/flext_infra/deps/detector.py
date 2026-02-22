from __future__ import annotations

from scripts.dependencies import detect_runtime_dev_deps as legacy


class RuntimeDevDetector:
    def run(self) -> int:
        return legacy.main()


def main() -> int:
    return RuntimeDevDetector().run()


if __name__ == "__main__":
    raise SystemExit(main())
