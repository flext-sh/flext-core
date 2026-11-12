"""Report what importing the flext-core root package loads, in a fresh interpreter.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys

from flext_core import __version__, __version_info__

facades = ("constants", "typings", "protocols", "models", "utilities")
loaded = [name for name in facades if f"flext_core.{name}" in sys.modules]
sys.stdout.write(f"{isinstance(__version__, str)} {loaded}\n")

from flext_core import u

sys.stdout.write(
    f"{u.PackageVersion.__name__} "
    f"{u.package_version_info(__version__) == __version_info__}\n"
)
