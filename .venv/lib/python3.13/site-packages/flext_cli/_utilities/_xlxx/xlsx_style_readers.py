"""Translate openpyxl visual styles into typed immutable models."""

from __future__ import annotations

from copy import copy

from openpyxl.styles import Alignment, Border, Color, Font, GradientFill
from openpyxl.styles.borders import Side
from openpyxl.styles.fills import Fill, PatternFill, Stop
from openpyxl.styles.styleable import StyleableObject

from flext_cli import m, p, r


class FlextCliUtilitiesXlsxStyleReaders:
    """Validate external style proxies into the canonical visual models."""

    # NOTE (multi-agent, mro-j2yt.1): copy() is the documented openpyxl proxy
    # boundary; each copied component is runtime-checked before model ingress.
    @staticmethod
    def _color_spec(color: Color | None) -> m.Cli.XlsxColor | None:
        if color is None:
            return None
        value = color.value
        if color.type == "rgb" and isinstance(value, str):
            return m.Cli.XlsxRgbColor(value=value, tint=color.tint)
        if color.type == "indexed" and isinstance(value, int):
            return m.Cli.XlsxIndexedColor(value=value, tint=color.tint)
        if color.type == "theme" and isinstance(value, int):
            return m.Cli.XlsxThemeColor(value=value, tint=color.tint)
        if color.type == "auto" and isinstance(value, bool):
            return m.Cli.XlsxAutomaticColor(tint=color.tint)
        msg = f"Unsupported openpyxl color: type={color.type!r}, value={value!r}"
        raise ValueError(msg)

    @classmethod
    def _font_spec(cls, font: Font) -> m.Cli.XlsxFontSpec:
        return m.Cli.XlsxFontSpec(
            name=font.name,
            size=font.size,
            bold=font.bold,
            italic=font.italic,
            strike=font.strike,
            outline=font.outline,
            shadow=font.shadow,
            condense=font.condense,
            extend=font.extend,
            underline=font.underline,
            vertical_align=font.vertAlign,
            color=cls._color_spec(font.color),
            charset=font.charset,
            family=font.family,
            scheme=font.scheme,
        )

    @classmethod
    def _fill_spec(cls, fill: Fill) -> m.Cli.XlsxFillSpec:
        if isinstance(fill, PatternFill):
            return m.Cli.XlsxPatternFillSpec(
                pattern=fill.patternType,
                foreground=cls._color_spec(fill.fgColor),
                background=cls._color_spec(fill.bgColor),
            )
        if isinstance(fill, GradientFill):
            stops = tuple(
                m.Cli.XlsxGradientStop(
                    color=cls._color_spec(item.color) or m.Cli.XlsxAutomaticColor(),
                    position=item.position,
                )
                for item in fill.stop
                if isinstance(item, Stop)
            )
            return m.Cli.XlsxGradientFillSpec(
                mode=fill.type,
                degree=fill.degree,
                left=fill.left,
                right=fill.right,
                top=fill.top,
                bottom=fill.bottom,
                stops=stops,
            )
        msg = f"Unsupported openpyxl fill: {fill.__class__.__name__}"
        raise TypeError(msg)

    @classmethod
    def _side_spec(cls, side: Side | None) -> m.Cli.XlsxBorderSideSpec | None:
        if side is None:
            return None
        return m.Cli.XlsxBorderSideSpec(
            style=side.style, color=cls._color_spec(side.color)
        )

    @classmethod
    def _border_spec(cls, border: Border) -> m.Cli.XlsxBorderSpec:
        return m.Cli.XlsxBorderSpec(
            left=cls._side_spec(border.left),
            right=cls._side_spec(border.right),
            top=cls._side_spec(border.top),
            bottom=cls._side_spec(border.bottom),
            start=cls._side_spec(border.start),
            end=cls._side_spec(border.end),
            diagonal=cls._side_spec(border.diagonal),
            vertical=cls._side_spec(border.vertical),
            horizontal=cls._side_spec(border.horizontal),
            diagonal_up=border.diagonalUp,
            diagonal_down=border.diagonalDown,
            outline=border.outline,
        )

    @staticmethod
    def _alignment_spec(alignment: Alignment) -> m.Cli.XlsxAlignmentSpec:
        return m.Cli.XlsxAlignmentSpec(
            horizontal=alignment.horizontal,
            vertical=alignment.vertical,
            text_rotation=alignment.textRotation or 0,
            wrap_text=alignment.wrapText,
            shrink_to_fit=alignment.shrinkToFit,
            indent=alignment.indent,
            relative_indent=alignment.relativeIndent,
            justify_last_line=alignment.justifyLastLine,
            reading_order=alignment.readingOrder,
        )

    @classmethod
    def _visual_from_styleable(
        cls, value: StyleableObject
    ) -> p.Result[m.Cli.XlsxVisualStyleSpec]:
        try:
            visual = cls._visual_from_styleable_unchecked(value)
        except (TypeError, m.ValidationError, ValueError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return r[m.Cli.XlsxVisualStyleSpec].fail(detail)
        return r[m.Cli.XlsxVisualStyleSpec].ok(visual)

    @classmethod
    def _visual_from_styleable_unchecked(
        cls, value: StyleableObject
    ) -> m.Cli.XlsxVisualStyleSpec:
        font = copy(value.font)
        fill = copy(value.fill)
        border = copy(value.border)
        alignment = copy(value.alignment)
        if not isinstance(font, Font):
            msg = "openpyxl font proxy copy is not Font"
            raise TypeError(msg)
        if not isinstance(fill, Fill):
            msg = "openpyxl fill proxy copy is not Fill"
            raise TypeError(msg)
        if not isinstance(border, Border):
            msg = "openpyxl border proxy copy is not Border"
            raise TypeError(msg)
        if not isinstance(alignment, Alignment):
            msg = "openpyxl alignment proxy copy is not Alignment"
            raise TypeError(msg)
        return m.Cli.XlsxVisualStyleSpec(
            font=cls._font_spec(font),
            fill=cls._fill_spec(fill),
            border=cls._border_spec(border),
            alignment=cls._alignment_spec(alignment),
            number_format=value.number_format,
        )


__all__: tuple[str, ...] = ("FlextCliUtilitiesXlsxStyleReaders",)
