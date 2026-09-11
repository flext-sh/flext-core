"""Round-trip YAML tree-editing helpers behind ``u.Cli.yaml_*``.

Comment/anchor-aware editing operations on parsed YAML trees: anchor clearing,
comment copying, pre-key comment insertion, in-place value updates, and
order-preserving overlays.

NOTE (multi-agent): mro-i6nq.13 — extracted from the removed
``_yaml_roundtrip_parts/..._part_02``. Composed into ``FlextCliUtilitiesYaml``
via MRO in ``yaml.py``; consumes the engine + convert mixins through the base
class. Do not duplicate these operations in a leaf module.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import ClassVar, TypeGuard

from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.tokens import CommentToken as RuamelCommentToken

from flext_cli import p, t
from flext_core import u

from ._engine import FlextCliUtilitiesYamlEngineMixin


class FlextCliUtilitiesYamlEditingMixin(FlextCliUtilitiesYamlEngineMixin):
    """Comment/anchor-aware editing operations on parsed YAML trees."""

    _module_logger: ClassVar[p.Logger] = u.fetch_logger(__name__)

    @staticmethod
    def _yaml_has_anchor(value: t.Cli.YamlValue) -> TypeGuard[p.Cli.YamlAnchorNode]:
        """Return True when a value exposes the ruamel anchor API."""
        return hasattr(value, "yaml_set_anchor")

    @staticmethod
    def yaml_clear_anchors(node: t.Cli.YamlValue) -> None:
        """Clear ruamel anchors recursively from a YAML value."""
        if node is None:
            return
        if FlextCliUtilitiesYamlEditingMixin._yaml_has_anchor(node):
            node.yaml_set_anchor(None)
        if isinstance(node, Mapping):
            for value in node.values():
                FlextCliUtilitiesYamlEditingMixin.yaml_clear_anchors(value)
        elif FlextCliUtilitiesYamlEditingMixin.yaml_is_sequence(node):
            for item in node:
                FlextCliUtilitiesYamlEditingMixin.yaml_clear_anchors(item)

    @staticmethod
    def yaml_deep_copy_comments(src: t.Cli.YamlNode, dst: t.Cli.YamlNode) -> None:
        """Copy ruamel comments from *src* to *dst* for commented containers."""
        if isinstance(src, CommentedMap) and isinstance(dst, CommentedMap):
            dst.ca.comment = src.ca.comment
            for key in src:
                if key in dst:
                    FlextCliUtilitiesYamlEditingMixin.yaml_deep_copy_comments(
                        src[key], dst[key]
                    )
        elif isinstance(src, CommentedSeq) and isinstance(dst, CommentedSeq):
            dst.ca.comment = src.ca.comment
            for index, item in enumerate(src):
                if index < len(dst):
                    FlextCliUtilitiesYamlEditingMixin.yaml_deep_copy_comments(
                        item, dst[index]
                    )

    @staticmethod
    def yaml_copy_key_comment(
        parent: CommentedMap, key: str, target: CommentedMap
    ) -> None:
        """Copy ruamel pre-key comments for one key between two maps."""
        if key in parent.ca.items:
            target.ca.items[key] = copy.deepcopy(parent.ca.items[key])

    @staticmethod
    def yaml_pre_key_tokens(node: CommentedMap, key: str) -> list[RuamelCommentToken]:
        """Return the existing pre-key comment tokens for one key."""
        existing = node.ca.items.get(key)
        if not existing or not existing[1]:
            return []
        post = existing[1]
        if isinstance(post, list):
            return [token for token in post if isinstance(token, RuamelCommentToken)]
        if isinstance(post, RuamelCommentToken):
            return [post]
        return []

    @staticmethod
    def _yaml_comment_core(text: str) -> str:
        """Reduce a comment to its framing-insensitive core text."""
        return text.lstrip("#").rstrip("\n").strip()

    @staticmethod
    def yaml_has_key_comment(node: CommentedMap, key: str, text: str) -> bool:
        r"""Return True when a pre-key comment with the same core text exists.

        NOTE (multi-agent): the comparison is framing-insensitive on BOTH
        sides ("# x", "# x\n" and "#x" all match the stored token "# x\n")
        because the write path normalizes through the same helper. This keeps
        ``yaml_add_pre_key_comment`` idempotent for any caller-supplied form;
        charts callers pass the full "# ...\n" token form and behave
        unchanged. Do NOT reintroduce a raw ``token.value == text`` comparison
        — it silently double-inserts for non-canonical input.
        """
        wanted = FlextCliUtilitiesYamlEditingMixin._yaml_comment_core(text)
        return any(
            FlextCliUtilitiesYamlEditingMixin._yaml_comment_core(token.value) == wanted
            for token in FlextCliUtilitiesYamlEditingMixin.yaml_pre_key_tokens(
                node, key
            )
        )

    @staticmethod
    def yaml_add_pre_key_comment(
        node: CommentedMap, key: str, text: str, path: tuple[str, ...] = ()
    ) -> None:
        """Insert one pre-key comment for a key. Idempotent for any input form."""
        if FlextCliUtilitiesYamlEditingMixin.yaml_has_key_comment(node, key, text):
            return
        comment_text = FlextCliUtilitiesYamlEditingMixin._yaml_comment_core(text)
        indent = max(len(path) - 1, 0) * 2
        node.yaml_set_comment_before_after_key(key, before=comment_text, indent=indent)

    @staticmethod
    def yaml_force_block_style(node: t.Cli.YamlNode) -> None:
        """Force block-style rendering for non-empty commented containers."""
        if isinstance(node, CommentedMap):
            if not node:
                return
            node.fa.set_block_style()
            for value in node.values():
                FlextCliUtilitiesYamlEditingMixin.yaml_force_block_style(value)
        elif isinstance(node, CommentedSeq):
            if not node:
                return
            node.fa.set_block_style()
            for item in node:
                FlextCliUtilitiesYamlEditingMixin.yaml_force_block_style(item)

    @staticmethod
    def yaml_update_value_inplace(
        node: CommentedMap, key: str, value: t.Cli.YamlValue
    ) -> None:
        """Update one key value, preserving its existing pre-key comments."""
        if isinstance(value, (dict, list)):
            node[key] = FlextCliUtilitiesYamlEditingMixin.yaml_deep_to_commented(value)
        else:
            node[key] = value

    @staticmethod
    def yaml_overlay_preserving_order(
        base: CommentedMap, overlay: Mapping[str, t.Cli.YamlValue] | CommentedMap
    ) -> None:
        """Overwrite *base* with *overlay*, preserving the original key order.

        New keys are appended at the end of the level so the pre-key comments
        of existing keys keep their position.
        """
        new_keys: list[tuple[str, t.Cli.YamlValue]] = []
        for key, value in overlay.items():
            if key in base:
                if isinstance(base[key], CommentedMap) and isinstance(value, dict):
                    FlextCliUtilitiesYamlEditingMixin.yaml_overlay_preserving_order(
                        base[key], value
                    )
                else:
                    FlextCliUtilitiesYamlEditingMixin.yaml_update_value_inplace(
                        base, key, value
                    )
            else:
                new_keys.append((
                    key,
                    FlextCliUtilitiesYamlEditingMixin.yaml_deep_to_commented(value),
                ))
        for key, value in new_keys:
            base[key] = value


__all__: list[str] = ["FlextCliUtilitiesYamlEditingMixin"]
