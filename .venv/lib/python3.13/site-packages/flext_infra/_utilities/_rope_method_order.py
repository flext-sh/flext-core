"""Method categorization + ordering for class-body reordering — extracted concern."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from flext_infra.constants import c

if TYPE_CHECKING:
    from flext_infra import m, t


class FlextInfraUtilitiesRopeMethodOrderMixin:
    """Match/sort/categorize methods against declarative ordering rules.

    Composed into FlextInfraUtilitiesRopeHelpers via inheritance; pure rule
    evaluation over ``m.Infra.MethodInfo``/``MethodOrderRule`` (no state).
    """

    _DECORATOR_TO_CATEGORY: ClassVar[t.StrPairSequence] = [
        ("staticmethod", "static"),
        ("classmethod", "class"),
    ]

    @staticmethod
    def matches_method_rule(
        method: m.Infra.MethodInfo, rule: m.Infra.MethodOrderRule
    ) -> bool:
        """Check if a method matches an ordering rule."""
        decorators = set(method.decorators)
        excludes = set(rule.exclude_decorators)
        match rule.visibility:
            case "public":
                visibility_matches = not method.name.startswith("_")
            case "protected":
                visibility_matches = method.name.startswith(
                    "_"
                ) and not method.name.startswith("__")
            case "private":
                visibility_matches = method.name.startswith(
                    "__"
                ) and not method.name.endswith("__")
            case _:
                visibility_matches = True
        decorators_match = not rule.decorators or bool(
            decorators.intersection(rule.decorators)
        )
        excluded = bool(excludes and decorators.intersection(excludes))
        patterns = rule.patterns
        patterns_match = not patterns or any(
            c.Infra.compile(pattern).match(method.name) for pattern in patterns
        )
        return (
            visibility_matches and decorators_match and patterns_match and not excluded
        )

    @staticmethod
    def build_method_sort_key(
        method: m.Infra.MethodInfo, order_config: t.SequenceOf[m.Infra.MethodOrderRule]
    ) -> tuple[int, int, str]:
        """Build a sort key tuple for method ordering."""
        cls = FlextInfraUtilitiesRopeMethodOrderMixin
        for index, rule in enumerate(order_config):
            if rule.category == "class_attributes":
                continue
            if not cls.matches_method_rule(method, rule):
                continue
            explicit_order = rule.order
            if explicit_order:
                if method.name in explicit_order:
                    return (index, explicit_order.index(method.name), method.name)
                if "*" in explicit_order:
                    return (index, explicit_order.index("*") + 1, method.name)
            return (index, 0, method.name)
        return (len(order_config), 0, method.name)

    @staticmethod
    def categorize_method(name: str, decorators: t.StrSequence) -> str:
        """Categorize a method by its decorators and name pattern."""
        result: str
        if c.Infra.PROPERTY_DECORATORS.intersection(decorators):
            result = c.Infra.MethodCategory.PROPERTY
        else:
            for (
                decorator_name,
                category,
            ) in FlextInfraUtilitiesRopeMethodOrderMixin._DECORATOR_TO_CATEGORY:
                if decorator_name in decorators:
                    result = category
                    break
            else:
                if name.startswith("__") and name.endswith("__"):
                    result = c.Infra.MethodCategory.MAGIC
                elif name.startswith("__"):
                    result = c.Infra.MethodCategory.PRIVATE
                elif name.startswith("_"):
                    result = c.Infra.MethodCategory.PROTECTED
                else:
                    result = c.Infra.MethodCategory.PUBLIC
        return result


__all__: list[str] = ["FlextInfraUtilitiesRopeMethodOrderMixin"]
