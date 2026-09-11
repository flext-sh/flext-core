"""Types extraction for flext_tests.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import m, t
from flext_tests._typings.base import FlextTestsBaseTypesMixin as tb


class FlextTestsFilesTypesMixin:
    type BatchFiles = (
        t.MappingKV[str, tb.TestobjectSerializable]
        | t.SequenceOf[tb.TestobjectSerializable]
    )
    type ReadContent = str | bytes | m.ConfigMap | t.SequenceOf[t.StrSequence]
    type FileContentStructured = (
        t.MappingKV[str, tb.TestobjectSerializable] | m.BaseModel
    )
    type FileContentPlain = ReadContent | FileContentStructured
