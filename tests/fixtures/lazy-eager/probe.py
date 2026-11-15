"""Report whether a deferred export is a stand-in, in a fresh interpreter.

Usage: ``probe.py <early|late|never> <python path> <capability module>``.
``early`` sets the eager flag before the export is read, ``late`` after.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys

from flext_core import install_lazy_exports, lazy

mode, path, module = sys.argv[1:4]
sys.path.insert(0, path)
if mode == "early":
    lazy.run_eager()
resolve, _ = install_lazy_exports(
    "lazy_eager_probe",
    {},
    {"Capability": (module, "Capability")},
    deferred_members={"Capability": ("VALUE",)},
)
value = resolve("Capability")
sys.stdout.write(f"{module in sys.modules} {type(vars(value)['VALUE']).__name__}\n")
if mode == "late":
    lazy.run_eager()
