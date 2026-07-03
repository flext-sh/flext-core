"""Mixins example aligned to stable result usage."""

from __future__ import annotations

from flext_core import r


def run() -> None:
    """Execute a simple mixin-like success flow."""
    result = r[str].ok("mixins-example")
    if not result.success:
        msg = "mixins example failed"
        raise RuntimeError(msg)


class Ex05FlextMixins:
    """Compatibility wrapper expected by examples package exports."""

    @staticmethod
    def run() -> None:
        """Run mixins example."""
        run()


if __name__ == "__main__":
    run()
