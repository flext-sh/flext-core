"""Test fixtures package for flext-core.

This package provides centralized test data fixtures for consistent testing
across all flext-core test modules using advanced Python 3.13 patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

# Export test fixture functions for conftest.py imports
from .error_scenarios import get_test_error_scenarios as get_test_error_scenarios
from .performance_data import (
    get_benchmark_data as get_benchmark_data,
)
from .performance_data import (
    get_performance_threshold as get_performance_threshold,
)
from .sample_data import (
    get_error_context as get_error_context,
)
from .sample_data import (
    get_sample_data as get_sample_data,
)
from .sample_data import (
    get_test_user_data as get_test_user_data,
)
from .test_constants import get_test_constants as get_test_constants
from .test_contexts import get_test_contexts as get_test_contexts
from .test_payloads import get_test_payloads as get_test_payloads
