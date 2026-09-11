"""Cohesive pyrefly-config fix-step mixin for FlextInfraConfigFixer.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING

from flext_core import r
from flext_infra import c, config, t, u
from flext_infra.deps.extra_paths import FlextInfraExtraPathsManager

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p


class FlextInfraConfigFixerSteps:
    """Mixin holding the three cohesive pyrefly fix-steps."""

    _repository_root: Path

    def _sync_search_path(
        self,
        pyrefly: MutableMapping[str, t.Infra.InfraValue],
        project_dir: Path,
        *,
        is_root: bool,
    ) -> p.Result[t.StrSequence]:
        """Synchronize tool.pyrefly.search-path from YAML rules."""
        search_raw = pyrefly.get(c.Infra.SEARCH_PATH)
        if not isinstance(search_raw, list):
            return r[t.StrSequence].ok(())
        try:
            current_paths: t.JsonList = t.Cli.JSON_LIST_ADAPTER.validate_python(
                list(search_raw)
            )
        except c.ValidationError as err:
            return r[t.StrSequence].fail_op("validate-search-path", err)
        current_search = [
            path_item for path_item in current_paths if isinstance(path_item, str)
        ]
        expected_search = FlextInfraExtraPathsManager(
            repository_root=self._repository_root
        ).pyrefly_search_paths(project_dir=project_dir, is_root=is_root)
        if current_search != expected_search:
            pyrefly[c.Infra.SEARCH_PATH] = u.Cli.toml_array(expected_search)
            return r[t.StrSequence].ok(["synchronized search-path from YAML rules"])
        return r[t.StrSequence].ok(())

    def _sync_project_includes(
        self,
        pyrefly: MutableMapping[str, t.Infra.InfraValue],
        project_dir: Path,
        *,
        is_root: bool,
    ) -> p.Result[t.StrSequence]:
        """Synchronize tool.pyrefly.project-includes from canonical path rules."""
        includes_raw = pyrefly.get(c.Infra.PROJECT_INCLUDES)
        if not isinstance(includes_raw, list):
            return r[t.StrSequence].ok(())
        try:
            current_items: t.JsonList = t.Cli.JSON_LIST_ADAPTER.validate_python(
                list(includes_raw)
            )
        except c.ValidationError as err:
            return r[t.StrSequence].fail_op("validate-project-includes", err)
        current_includes = [
            path_item for path_item in current_items if isinstance(path_item, str)
        ]
        expected_includes = FlextInfraExtraPathsManager(
            repository_root=self._repository_root
        ).pyrefly_project_includes(project_dir=project_dir, is_root=is_root)
        if current_includes != expected_includes:
            pyrefly[c.Infra.PROJECT_INCLUDES] = u.Cli.toml_array(expected_includes)
            return r[t.StrSequence].ok([
                "synchronized project-includes from YAML rules"
            ])
        return r[t.StrSequence].ok(())

    def _strip_ignored_sub_configs(
        self, pyrefly: MutableMapping[str, t.Infra.InfraValue]
    ) -> p.Result[tuple[t.StrSequence, bool]]:
        """Drop ignore=true entries from tool.pyrefly.sub-config."""
        sub_configs = pyrefly.get(c.Infra.SUB_CONFIG)
        if not isinstance(sub_configs, list):
            return r[tuple[t.StrSequence, bool]].ok(((), False))
        try:
            configs: t.SequenceOf[t.Infra.InfraValue] = (
                t.Infra.INFRA_SEQ_ADAPTER.validate_python(sub_configs)
            )
        except c.ValidationError as err:
            return r[tuple[t.StrSequence, bool]].fail_op("validate-sub-configs", err)
        fixes: t.MutableSequenceOf[str] = []
        removed_ignore = False
        new_configs: t.MutableSequenceOf[t.Infra.InfraValue] = []
        for conf in configs:
            conf_out: t.Infra.InfraValue = conf
            if isinstance(conf, Mapping):
                try:
                    conf_map = t.Infra.INFRA_MAPPING_ADAPTER.validate_python(conf)
                    conf_out = dict(conf_map)
                except c.ValidationError as err:
                    return r[tuple[t.StrSequence, bool]].fail_op(
                        "validate-pyrefly-sub-config", err
                    )
            else:
                new_configs.append(conf_out)
                continue
            if conf_map.get(c.Infra.IGNORE) is True:
                removed_ignore = True
                matches = conf_map.get("matches", c.Infra.DEFAULT_UNKNOWN)
                fixes.append(f"removed ignore=true sub-settings for '{matches}'")
                continue
            new_configs.append(conf_out)
        if len(new_configs) != len(configs):
            pyrefly[c.Infra.SUB_CONFIG] = list(
                t.Cli.JSON_LIST_ADAPTER.validate_python(new_configs)
            )
        return r[tuple[t.StrSequence, bool]].ok((fixes, removed_ignore))

    def _sync_project_excludes(
        self, pyrefly: MutableMapping[str, t.Infra.InfraValue]
    ) -> p.Result[t.StrSequence]:
        """Synchronize tool.pyrefly.project-excludes from YAML rules."""
        current_excludes: t.StrSequence = []
        excludes = pyrefly.get(c.Infra.PROJECT_EXCLUDES)
        if isinstance(excludes, list):
            try:
                exclude_items: t.JsonList = t.Cli.JSON_LIST_ADAPTER.validate_python([
                    *excludes
                ])
            except c.ValidationError as err:
                return r[t.StrSequence].fail_op("validate-project-excludes", err)
            current_excludes = [str(value) for value in exclude_items]
        expected_excludes = sorted(
            set(config.Infra.tooling.tools.pyrefly.project_exclude_globs)
        )
        if current_excludes != expected_excludes:
            pyrefly[c.Infra.PROJECT_EXCLUDES] = u.Cli.toml_array(expected_excludes)
            return r[t.StrSequence].ok([
                "synchronized project-excludes from YAML rules"
            ])
        return r[t.StrSequence].ok(())


__all__: list[str] = ["FlextInfraConfigFixerSteps"]
