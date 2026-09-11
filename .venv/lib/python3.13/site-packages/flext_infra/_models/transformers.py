"""Domain models for the transformers subpackage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar

from flext_cli import m
from flext_infra import t
from flext_infra._models._defaults import ImmutableEmptyMapping


class FlextInfraModelsTransformers:
    """Models for source transformers — exposed through the ``m.Infra`` facade."""

    class AliasMigrationContext(m.ContractModel):
        """Resolved ownership and import-root context for one alias migration."""

        policy_owner: Annotated[
            str,
            m.Field(description="Project package that owns the canonical alias policy"),
        ]
        import_root: Annotated[
            str, m.Field(description="Public facade root from which consumers import")
        ]

    class AliasMigrationEdit(m.ContractModel):
        """One validated in-memory canonical alias source rewrite."""

        # Why (flext-ygc2k): source bytes must survive validation byte-exact;
        # the strict base strips whitespace, which corrupts CAS comparisons.
        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(str_strip_whitespace=False)

        file_path: Annotated[Path, m.Field(description="Source file to rewrite")]
        original_source: Annotated[
            str, m.Field(description="Source bytes before migration")
        ]
        updated_source: Annotated[
            str, m.Field(description="Prospective source bytes after migration")
        ]
        changes: Annotated[
            tuple[str, ...], m.Field(description="Recorded migration operations")
        ] = ()

    class Tier0ImportAnalysis(m.Value):
        """Detection results for a single Python file self-import patterns."""

        # Why: value contract owned by m.Infra transformers facet, not nested in the fixer service.
        package_name: Annotated[
            str, m.Field(description="Resolved package name for the analyzed file")
        ]
        file_path: Annotated[
            Path,
            m.Field(description="Python file analyzed for Tier 0 import violations"),
        ]
        alias_to_module: Annotated[
            t.StrMapping,
            m.Field(description="Alias names mapped to their source modules"),
        ] = m.Field(default_factory=ImmutableEmptyMapping)
        category_a: Annotated[
            frozenset[str],
            m.Field(description="Top-level aliases that are informational only"),
        ] = m.Field(default_factory=frozenset)
        category_b: Annotated[
            frozenset[str],
            m.Field(description="Core aliases to redirect to the core package"),
        ] = m.Field(default_factory=frozenset)
        category_c: Annotated[
            frozenset[str],
            m.Field(description="Aliases to move into a TYPE_CHECKING block"),
        ] = m.Field(default_factory=frozenset)
        category_d: Annotated[
            frozenset[str],
            m.Field(
                description="Runtime-used aliases requiring direct import handling"
            ),
        ] = m.Field(default_factory=frozenset)

        @m.computed_field
        @property
        def has_violations(self) -> bool:
            """True if any imports need redirecting or moving."""
            return bool(self.category_b or self.category_c or self.category_d)

    class SourceRewrite(m.ArbitraryTypesModel):
        """One source rewrite: replace ``source[start:end]`` with ``text``."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        start: Annotated[int, m.Field(description="Start byte offset in the source")]
        end: Annotated[int, m.Field(description="End byte offset in the source")]
        text: Annotated[str, m.Field(description="Replacement text")]

    class HeaderSpan(m.ArbitraryTypesModel):
        """Byte offsets for the logical sections of a Python module header.

        Mutable accumulator populated incrementally by the header tokenizer
        (``transformers/_header.py``); never frozen while a parse is open.
        """

        shebang_end: Annotated[
            int, m.Field(description="Byte offset just after the shebang line")
        ] = 0
        encoding_end: Annotated[
            int, m.Field(description="Byte offset just after the encoding cookie")
        ] = 0
        comments_end: Annotated[
            int, m.Field(description="Byte offset after the leading comment block")
        ] = 0
        docstring_end: Annotated[
            int, m.Field(description="Byte offset after the module docstring")
        ] = 0
        last_import_end: Annotated[
            int, m.Field(description="Byte offset after the last import statement")
        ] = 0

    class HeaderInfo(m.ArbitraryTypesModel):
        """Structural summary of a module header."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        has_future_annotations: Annotated[
            bool, m.Field(description="Whether the module already imports annotations")
        ]
        aliases: Annotated[
            frozenset[str],
            m.Field(description="Local names bound by from-import statements"),
        ]
        span: Annotated[
            FlextInfraModelsTransformers.HeaderSpan,
            m.Field(description="Byte offsets for the header sections"),
        ]


__all__: list[str] = ["FlextInfraModelsTransformers"]
