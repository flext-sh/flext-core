"""Automation utilities for quad API support (model/dict/kwargs/hybrid).

This module provides the core building blocks for implementing flexible APIs
that accept configuration in multiple formats without duplicating validation logic.

Part of Phase 1: Automation Core + Helpers
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def coerce_model(
    model_class: type[T],
    options: T | dict[str, Any] | None = None,
    **kwargs: object,
) -> T:
    """Universal model coercion supporting quad API.

    Accepts configuration in 4 ways:
    1. Model instance (passthrough)
    2. Dict (convert to model)
    3. kwargs (construct model)
    4. Hybrid (merge options dict + kwargs)

    Args:
        model_class: Target Pydantic model class
        options: Pre-existing model or dict
        **kwargs: Additional fields (merged with options if dict)

    Returns:
        Validated model instance

    Raises:
        ValidationError: If validation fails (Pydantic error)
        TypeError: If options type is invalid

    Examples:
        >>> from pydantic import BaseModel, Field
        >>> class Config(BaseModel):
        ...     value: int = Field(ge=0)
        ...     name: str = "default"
        >>>
        >>> # Style 1: Model instance (passthrough)
        >>> opts = coerce_model(Config, Config(value=5, name="test"))
        >>> assert opts.value == 5
        >>>
        >>> # Style 2: Dict
        >>> opts = coerce_model(Config, {"value": 5, "name": "test"})
        >>> assert opts.value == 5
        >>>
        >>> # Style 3: kwargs only
        >>> opts = coerce_model(Config, value=5, name="test")
        >>> assert opts.value == 5
        >>>
        >>> # Style 4: Hybrid (dict + kwargs, kwargs override)
        >>> opts = coerce_model(Config, {"value": 5}, name="override")
        >>> assert opts.name == "override"

    """
    # Case 1: Already a model instance
    if isinstance(options, model_class):
        if kwargs:
            # Hybrid: model + kwargs (create new instance with overrides)
            data = options.model_dump()
            data.update(kwargs)
            return model_class(**data)
        return options

    # Case 2: Dict or hybrid dict+kwargs
    if isinstance(options, dict):
        data = {**options, **kwargs}
        return model_class(**data)

    # Case 3: Pure kwargs (options is None)
    if options is None:
        return model_class(**kwargs)

    # Case 4: Invalid type
    model_name = getattr(model_class, "__name__", "Model")
    options_type = type(options).__name__
    msg = f"options must be {model_name}, dict, or None; got {options_type}"
    raise TypeError(msg)


__all__ = ["coerce_model"]
