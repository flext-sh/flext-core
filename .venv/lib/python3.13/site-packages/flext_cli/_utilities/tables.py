"""CLI table data helpers shared through ``u.Cli``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar

from tabulate import tabulate

from flext_cli import c, m, p, r, t
from flext_core import u


class FlextCliUtilitiesTables:
    """Table helpers exposed through ``u.Cli.tables_*``."""

    TABLE_DATA_ADAPTER: ClassVar[t.ValueAdapter[t.Cli.TableDataSource]] = m.TypeAdapter(
        t.Cli.TableDataSource
    )

    @staticmethod
    def tables_normalize_mapping_row(
        row: t.Cli.TableMappingRow,
    ) -> t.Cli.TableMappingRow:
        """Normalize one mapping row to JSON-compatible values."""
        return {key: u.normalize_to_json_value(value) for key, value in row.items()}

    @staticmethod
    def tables_normalize_sequence_row(
        row: t.Cli.TableSequenceRow,
    ) -> t.Cli.TableSequenceRow:
        """Normalize one sequence row to JSON-compatible values."""
        return [u.normalize_to_json_value(value) for value in row]

    @staticmethod
    def tables_resolve_config(
        settings: m.Cli.TableConfig | None = None,
        **settings_kwargs: t.Cli.TableConfigValue,
    ) -> p.Result[m.Cli.TableConfig]:
        """Resolve table config via canonical Pydantic model contract."""
        try:
            if settings is not None and not settings_kwargs:
                return r[m.Cli.TableConfig].ok(settings)
            base_data = (
                settings.model_dump(exclude_computed_fields=True)
                if settings is not None
                else {}
            )
            settings_data = {**base_data, **settings_kwargs}
            resolved = m.Cli.TableConfig.model_validate(settings_data)
            return r[m.Cli.TableConfig].ok(resolved)
        except c.Cli.CLI_SAFE_EXCEPTIONS as exc:
            return r[m.Cli.TableConfig].fail(
                c.Cli.OUTPUT_TABLE_CONFIG_INVALID_FMT.format(error=exc)
            )

    @staticmethod
    def tables_normalize_data(
        data: t.Cli.TableDataSource,
    ) -> p.Result[Sequence[t.Cli.TableRow]]:
        """Validate and normalize mapping/sequence inputs to tabulate rows."""
        try:
            validated_data = FlextCliUtilitiesTables.TABLE_DATA_ADAPTER.validate_python(
                data
            )
        except c.ValidationError as exc:
            return r[Sequence[t.Cli.TableRow]].fail(
                c.Cli.OUTPUT_TABLE_DATA_INVALID_FMT.format(error=exc)
            )

        if isinstance(validated_data, Mapping):
            validated_mapping = validated_data
            return r[Sequence[t.Cli.TableRow]].ok([
                {"Key": key, "Value": u.normalize_to_json_value(value)}
                for key, value in validated_mapping.items()
            ])

        normalized_rows: t.MutableSequenceOf[t.Cli.TableRow] = []
        for row in validated_data:
            if isinstance(row, Mapping):
                normalized_rows.append(
                    FlextCliUtilitiesTables.tables_normalize_mapping_row(row)
                )
                continue
            if not isinstance(row, str):
                normalized_rows.append(
                    FlextCliUtilitiesTables.tables_normalize_sequence_row(row)
                )
                continue
            return r[Sequence[t.Cli.TableRow]].fail(c.Cli.OUTPUT_TABLE_ROW_INVALID)

        return r[Sequence[t.Cli.TableRow]].ok(normalized_rows)

    @staticmethod
    def tables_tabulate_payload(
        rows: t.SequenceOf[t.Cli.TableRow], headers: str | t.StrSequence
    ) -> tuple[
        t.SequenceOf[t.Cli.TableRow] | t.SequenceOf[t.Cli.TableSequenceRow],
        str | t.StrSequence,
    ]:
        """Build table data/header values accepted by tabulate."""
        table_data: (
            t.SequenceOf[t.Cli.TableRow] | t.SequenceOf[t.Cli.TableSequenceRow]
        ) = rows
        table_headers: str | t.StrSequence = headers
        if rows and isinstance(rows[0], Mapping) and not isinstance(headers, str):
            table_data = [
                list(row.values()) for row in rows if isinstance(row, Mapping)
            ]
            table_headers = list(headers)
        return table_data, table_headers

    @staticmethod
    def tables_render(
        rows: t.SequenceOf[t.Cli.TableRow], settings: m.Cli.TableConfig
    ) -> p.Result[str]:
        """Render normalized rows to a tabulated string."""
        headers: str | t.StrSequence
        if not settings.show_header:
            # NOTE (multi-agent): Empty headers use the immutable sequence contract.
            headers = ()
        elif isinstance(settings.headers, str):
            headers = settings.headers
        else:
            headers = tuple(settings.headers)

        colalign = settings.colalign
        if isinstance(headers, str):
            if not rows:
                column_count = 0
            elif isinstance(rows[0], Mapping):
                column_count = len(rows[0])
            else:
                column_count = len(rows[0])
        else:
            column_count = len(headers)

        if colalign is not None and column_count > 0 and len(colalign) > column_count:
            colalign = colalign[:column_count]

        table_data, table_headers = FlextCliUtilitiesTables.tables_tabulate_payload(
            rows, headers
        )
        try:
            rendered_table = tabulate(
                table_data,
                headers=table_headers,
                tablefmt=settings.table_backend_format,
                floatfmt=settings.floatfmt,
                numalign=settings.numalign,
                stralign=settings.stralign,
                missingval=settings.missingval,
                showindex=settings.showindex,
                disable_numparse=settings.disable_numparse,
                colalign=colalign,
            )
            return r[str].ok(rendered_table)
        except c.Cli.CLI_SAFE_EXCEPTIONS as exc:
            return r[str].fail_op(c.Cli.OUTPUT_TABLE_FORMATTING_OPERATION, exc)


__all__: t.MutableSequenceOf[str] = ["FlextCliUtilitiesTables"]
