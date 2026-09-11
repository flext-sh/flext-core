"""Translate typed visual style models into openpyxl objects."""

from __future__ import annotations

from openpyxl.styles import Alignment, Border, Color, Font, GradientFill, NamedStyle
from openpyxl.styles.borders import Side
from openpyxl.styles.fills import PatternFill, Stop

from flext_cli import m


class FlextCliUtilitiesXlsxStyleBuilders:
    """Build external visual styles only from validated model fields."""

    # NOTE (multi-agent, mro-j2yt.1): protection is deliberately absent from
    # every builder and is applied only by the worksheet protection adapter.
    @staticmethod
    def _color(spec: m.Cli.XlsxColor | None) -> Color | None:
        if spec is None:
            return None
        if spec.kind == "rgb":
            return Color(rgb=spec.value, tint=spec.tint)
        if spec.kind == "indexed":
            return Color(indexed=spec.value, tint=spec.tint)
        if spec.kind == "theme":
            return Color(theme=spec.value, tint=spec.tint)
        return Color(auto=True, tint=spec.tint)

    @classmethod
    def _font(cls, spec: m.Cli.XlsxFontSpec) -> Font:
        return Font(
            name=spec.name,
            size=spec.size,
            bold=spec.bold,
            italic=spec.italic,
            strikethrough=spec.strike,
            outline=spec.outline,
            shadow=spec.shadow,
            condense=spec.condense,
            extend=spec.extend,
            underline=spec.underline,
            vertAlign=spec.vertical_align,
            color=cls._color(spec.color),
            charset=spec.charset,
            family=spec.family,
            scheme=spec.scheme,
        )

    @classmethod
    def _fill(cls, spec: m.Cli.XlsxFillSpec) -> PatternFill | GradientFill:
        if spec.kind == "pattern":
            foreground = cls._color(spec.foreground)
            background = cls._color(spec.background)
            return PatternFill(
                patternType=spec.pattern,
                fgColor=foreground if foreground is not None else Color(),
                bgColor=background if background is not None else Color(),
            )
        stops = tuple(
            Stop(color=cls._color(item.color) or Color(), position=item.position)
            for item in spec.stops
        )
        return GradientFill(
            type=spec.mode,
            degree=spec.degree,
            left=spec.left,
            right=spec.right,
            top=spec.top,
            bottom=spec.bottom,
            stop=stops,
        )

    @classmethod
    def _side(cls, spec: m.Cli.XlsxBorderSideSpec | None) -> Side | None:
        if spec is None:
            return None
        return Side(style=spec.style, color=cls._color(spec.color))

    @classmethod
    def _border(cls, spec: m.Cli.XlsxBorderSpec) -> Border:
        return Border(
            left=cls._side(spec.left),
            right=cls._side(spec.right),
            top=cls._side(spec.top),
            bottom=cls._side(spec.bottom),
            start=cls._side(spec.start),
            end=cls._side(spec.end),
            diagonal=cls._side(spec.diagonal),
            vertical=cls._side(spec.vertical),
            horizontal=cls._side(spec.horizontal),
            diagonalUp=spec.diagonal_up,
            diagonalDown=spec.diagonal_down,
            outline=spec.outline,
        )

    @staticmethod
    def _alignment(spec: m.Cli.XlsxAlignmentSpec) -> Alignment:
        return Alignment(
            horizontal=spec.horizontal,
            vertical=spec.vertical,
            textRotation=spec.text_rotation,
            wrapText=spec.wrap_text,
            shrinkToFit=spec.shrink_to_fit,
            indent=spec.indent,
            relativeIndent=spec.relative_indent,
            justifyLastLine=spec.justify_last_line,
            readingOrder=spec.reading_order,
        )

    @classmethod
    def _named_style(cls, spec: m.Cli.XlsxNamedStyleSpec) -> NamedStyle:
        visual = spec.visual
        return NamedStyle(
            name=spec.name,
            font=cls._font(visual.font),
            fill=cls._fill(visual.fill),
            border=cls._border(visual.border),
            alignment=cls._alignment(visual.alignment),
            number_format=visual.number_format,
        )


__all__: tuple[str, ...] = ("FlextCliUtilitiesXlsxStyleBuilders",)
