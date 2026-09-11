"""File-comparison parsing helpers for FlextTestsFiles."""

from __future__ import annotations

from flext_core import r
from flext_tests import c, m, t, u
from flext_tests._utilities._files._creation import FlextTestsFilesCreationMixin
from flext_tests._utilities.payload import FlextTestsPayloadUtilities
from flext_tests.protocols import p


class FlextTestsFilesComparisonMixin:
    """Compare two files by content, lines, size, hash, or deep structure."""

    type ParsedPair = tuple[
        t.MappingKV[str, t.Tests.TestobjectSerializable],
        t.MappingKV[str, t.Tests.TestobjectSerializable],
    ]

    @staticmethod
    def _read_both(params: m.Tests.CompareParams) -> t.StrPair:
        enc = c.Tests.DEFAULT_ENCODING
        return (
            params.file1.read_text(encoding=enc),
            params.file2.read_text(encoding=enc),
        )

    def _try_parse_both(
        self, content1: str, content2: str, fmt: str
    ) -> p.Result[FlextTestsFilesComparisonMixin.ParsedPair]:
        """Try to parse both contents as dicts in given format."""
        parse = (
            u.Cli.json_parse
            if fmt == "json"
            else u.Cli.yaml_parse
            if fmt == "yaml"
            else None
        )
        if parse is None:
            return r[FlextTestsFilesComparisonMixin.ParsedPair].fail(
                f"unsupported comparison format: {fmt}"
            )
        parsed_result = u.try_(
            lambda: (parse(content1), parse(content2)),
            catch=(ValueError, c.Cli.YamlParseError, TypeError),
            op_name="parse comparison contents",
        )
        if parsed_result.failure:
            return r[FlextTestsFilesComparisonMixin.ParsedPair].fail(
                parsed_result.error or "parse comparison contents failed"
            )
        r1, r2 = parsed_result.value
        d1 = r1.value if r1.success else None
        d2 = r2.value if r2.success else None
        if FlextTestsFilesCreationMixin.is_mapping(
            d1
        ) and FlextTestsFilesCreationMixin.is_mapping(d2):
            return r[FlextTestsFilesComparisonMixin.ParsedPair].ok((
                FlextTestsFilesCreationMixin.to_payload_mapping(d1),
                FlextTestsFilesCreationMixin.to_payload_mapping(d2),
            ))
        return r[FlextTestsFilesComparisonMixin.ParsedPair].fail(
            "comparison contents are not both mappings"
        )

    def _apply_key_filtering(
        self,
        dict1: t.MappingKV[str, t.Tests.TestobjectSerializable],
        dict2: t.MappingKV[str, t.Tests.TestobjectSerializable],
        keys: t.StrSequence | None,
        exclude_keys: t.StrSequence | None,
    ) -> tuple[
        t.MappingKV[str, t.Tests.TestobjectSerializable],
        t.MappingKV[str, t.Tests.TestobjectSerializable],
    ]:
        """Apply key filtering to both dicts if specified."""
        if keys is None and exclude_keys is None:
            return (dict1, dict2)
        filter_keys_set = set(keys) if keys is not None else None
        exclude_keys_set = set(exclude_keys) if exclude_keys is not None else None
        result1 = u.transform(
            FlextTestsPayloadUtilities.to_config_map(dict1),
            filter_keys=filter_keys_set,
            exclude_keys=exclude_keys_set,
        )
        result2 = u.transform(
            FlextTestsPayloadUtilities.to_config_map(dict2),
            filter_keys=filter_keys_set,
            exclude_keys=exclude_keys_set,
        )
        if result1.success and result2.success:
            filtered1 = FlextTestsFilesCreationMixin.to_payload_mapping(result1.value)
            filtered2 = FlextTestsFilesCreationMixin.to_payload_mapping(result2.value)
            return (filtered1, filtered2)
        return (dict1, dict2)


__all__: list[str] = ["FlextTestsFilesComparisonMixin"]
