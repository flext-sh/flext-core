"""Integration fixtures for enforcement validator.

Real modules that exercise every rule in ``c.ENFORCEMENT_RULES`` — imported
by ``test_enforcement_integration.py`` to validate that the ``__pydantic_init_subclass__``
hook actually fires the expected warnings on real code.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""
