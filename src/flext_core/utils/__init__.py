"""FLEXT Core Utilities Package.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Enterprise utilities for FLEXT framework.
"""

from __future__ import annotations

import contextlib

# Optional imports with deprecation warnings
with contextlib.suppress(ImportError):
    from flext_core.utils.ldif_writer import FlextLDIFWriter
    from flext_core.utils.ldif_writer import LDIFHierarchicalSorter
    from flext_core.utils.ldif_writer import LDIFWriter

# Configuration generator for eliminating duplicate scripts
with contextlib.suppress(ImportError):
    from flext_core.utils.config_generator import BaseConfigGenerator
    from flext_core.utils.config_generator import ConfigGeneratorFactory
    from flext_core.utils.config_generator import ConfigSection
    from flext_core.utils.config_generator import ProjectType
    from flext_core.utils.config_generator import generate_project_config

# Initialize __all__
__all__: list[str] = []

# Add deprecated LDIF exports if available
if "FlextLDIFWriter" in locals():
    __all__ += ["FlextLDIFWriter", "LDIFHierarchicalSorter", "LDIFWriter"]

# Add config generator exports if available
if "BaseConfigGenerator" in locals():
    __all__ += [
        "BaseConfigGenerator",
        "ConfigGeneratorFactory",
        "ConfigSection",
        "ProjectType",
        "generate_project_config",
    ]
