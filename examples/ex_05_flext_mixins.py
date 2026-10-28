"""Result success check for the mixins-named example entry point.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

from examples import r


class ExamplesFlextMixins:
    """Expose the example's result success check through a class entry point."""

    @staticmethod
    def run() -> None:
        """Check a successful result without exercising mixin behavior.

        Raises:
            RuntimeError: If the result success check fails.

        """
        result = r[str].ok("mixins-example")
        if not result.success:
            msg = "mixins example failed"
            raise RuntimeError(msg)


if __name__ == "__main__":
    ExamplesFlextMixins.run()

__all__ = ["ExamplesFlextMixins"]
