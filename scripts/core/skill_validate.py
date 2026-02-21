#!/usr/bin/env python3
# Owner-Skill: .claude/skills/scripts-infra/SKILL.md
"""Data-driven skill validator for rules.yml-based policy gates."""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
import time
from pathlib import Path

_SCRIPTS_ROOT = str(Path(__file__).resolve().parents[1])
if _SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, _SCRIPTS_ROOT)

from libs.discovery import discover_projects as ssot_discover_projects  # noqa: E402

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:
    yaml = None  # type: ignore[assignment]


SKILLS_DIR = Path(".claude/skills")
REPORT_DEFAULT = ".claude/skills/{skill}/report.json"
BASELINE_DEFAULT = ".claude/skills/{skill}/baseline.json"
CACHE_TTL_SECONDS = 300

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_INFRA = 3


class SkillUsageError(Exception):
    """Raise when rules.yml or CLI usage is invalid."""


class SkillInfraError(Exception):
    """Raise when external tools or filesystem operations fail."""


def eprint(message: str) -> None:
    """Print an error message to stderr."""
    print(message, file=sys.stderr)


def run_cmd(
    command: list[str],
    cwd: Path,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    """Run a command and normalize infrastructure failures."""
    try:
        return subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        msg = f"Command not found: {command[0]}"
        raise SkillInfraError(msg) from exc
    except subprocess.TimeoutExpired as exc:
        msg = f"Command timed out: {' '.join(command[:3])}"
        raise SkillInfraError(msg) from exc


def tool_available(name: str) -> bool:
    """Return whether a CLI tool is available in PATH."""
    try:
        result = subprocess.run(
            [name, "--version"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode in {0, 1}


def normalize_rel_path(value: str) -> str:
    """Normalize path separators and leading relative markers."""
    return value.replace("\\", "/").lstrip("./")


def normalize_exclude_globs(
    excludes: list[str],
    known_projects: set[str],
) -> list[str]:
    """Strip project-name prefixes from exclude globs.

    Exclude patterns like ``flext-core/src/foo.py`` never match tracked
    files which are project-relative (``src/foo.py``).
    """
    result: list[str] = []
    for pat in excludes:
        stripped = pat
        for proj in known_projects:
            prefix = proj + "/"
            if pat.startswith(prefix):
                stripped = pat[len(prefix) :]
                print(
                    f"  Warning: exclude '{pat}' stripped to '{stripped}' (project-relative)",
                    file=sys.stderr,
                )
                break
        result.append(stripped)
    return result


def unique_sorted(items: list[str]) -> list[str]:
    """Return sorted unique values preserving first-seen semantics."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return sorted(out)


def load_rules_yml(path: Path) -> dict[str, object]:
    """Load rules metadata from YAML or JSON fallback."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"Cannot read rules file: {path}"
        raise SkillInfraError(msg) from exc

    safe_load = getattr(yaml, "safe_load", None)
    if safe_load is None:
        msg = "PyYAML safe_load is unavailable"
        raise SkillInfraError(msg)

    try:
        parsed = safe_load(raw)
    except Exception as exc:
        msg = f"Invalid YAML at {path}: {exc}"
        raise SkillInfraError(msg) from exc
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        msg = f"rules.yml must be a mapping: {path}"
        raise SkillUsageError(msg)
    return dict(parsed)


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write a JSON payload to disk with stable formatting."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        msg = f"Cannot write JSON file: {path}"
        raise SkillInfraError(msg) from exc


def read_json(path: Path) -> dict[str, object]:
    """Read a JSON object from disk or return an empty mapping."""
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Cannot parse JSON file: {path}"
        raise SkillInfraError(msg) from exc
    if not isinstance(parsed, dict):
        msg = f"JSON file must be an object: {path}"
        raise SkillInfraError(msg)
    return dict(parsed)


def render_path_template(
    root: Path,
    template: str,
    skill: str,
    fallback: str,
) -> Path:
    """Render a skill path template into an absolute filesystem path."""
    value = template or fallback
    rendered = value.replace("{skill}", skill)
    candidate = Path(rendered)
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def discover_projects(root: Path) -> dict[str, object]:
    """Discover workspace projects, matching Makefile logic.

    Returns:
        {"flext": ["flext-core", "flext-api", ...],
         "external": ["flext-oud-mig", ...],
         "root": "."}

    """
    discovered = ssot_discover_projects(root)
    flext_projects = [
        project.name for project in discovered if project.kind == "submodule"
    ]
    external_projects = [
        project.name for project in discovered if project.kind == "external"
    ]

    return {
        "flext": unique_sorted(flext_projects),
        "external": unique_sorted(external_projects),
        "root": ".",
    }


def build_project_lookup(
    root: Path,
    discovered: dict[str, object],
) -> dict[str, Path]:
    """Build a lookup table from project names to absolute paths."""
    lookup: dict[str, Path] = {".": root.resolve()}

    flext_values = discovered.get("flext", [])
    external_values = discovered.get("external", [])

    if isinstance(flext_values, list):
        for name in flext_values:
            if isinstance(name, str):
                lookup[name] = (root / name).resolve()

    if isinstance(external_values, list):
        for name in external_values:
            if isinstance(name, str):
                lookup[name] = (root / name).resolve()

    return lookup


_git_files_cache: dict[str, tuple[float, list[str]]] = {}


def get_tracked_files(project_path: Path) -> list[str]:
    """Get git-tracked files for a project via git ls-files (in-memory cached)."""
    key = str(project_path.resolve())
    now = time.monotonic()
    cached = _git_files_cache.get(key)
    if cached and (now - cached[0]) < CACHE_TTL_SECONDS:
        return cached[1]

    if not project_path.exists():
        msg = f"Project path does not exist: {project_path}"
        raise SkillUsageError(msg)

    result = run_cmd(["git", "ls-files"], cwd=project_path, timeout=120)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        msg = f"git ls-files failed in {project_path}: {stderr}"
        raise SkillInfraError(msg)

    files = unique_sorted([
        normalize_rel_path(line)
        for line in (result.stdout or "").splitlines()
        if line.strip()
    ])

    _git_files_cache[key] = (now, files)
    return files


def filter_files(
    tracked_files: list[str],
    include_globs: list[str],
    exclude_globs: list[str],
    project_path: Path | None = None,
) -> list[str]:
    """Filter tracked files using include and exclude glob patterns."""
    includes = include_globs or ["**/*"]
    excludes = exclude_globs or []

    selected: list[str] = []
    for rel in tracked_files:
        rel_norm = normalize_rel_path(rel)
        if not any(fnmatch.fnmatch(rel_norm, pattern) for pattern in includes):
            continue
        if any(fnmatch.fnmatch(rel_norm, pattern) for pattern in excludes):
            continue
        # Skip files that don't exist on disk (e.g. stale submodule entries)
        if project_path is not None and not (project_path / rel_norm).exists():
            continue
        selected.append(rel_norm)
    return unique_sorted(selected)


def chunk_list(items: list[str], size: int) -> list[list[str]]:
    """Split a list into fixed-size chunks."""
    if size <= 0:
        return [items]
    out: list[list[str]] = [
        items[idx : idx + size] for idx in range(0, len(items), size)
    ]
    return out


def parse_count_from_json_lines(output: str) -> int:
    """Sum violation counts from line-delimited JSON output."""
    total = 0
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        maybe_count = payload.get("violation_count", payload.get("count", 0))
        if isinstance(maybe_count, int):
            total += maybe_count
    return total


def run_ast_grep_rule(
    *,
    rule: dict[str, object],
    skill_dir: Path,
    project_name: str,
    project_path: Path,
    include_globs: list[str],
    exclude_globs: list[str],
    allowed_files: set[str],
) -> tuple[dict[str, int], int]:
    """Execute one ast-grep rule and aggregate matched counts."""
    if not tool_available("sg"):
        msg = "ast-grep (sg) is required but not available"
        raise SkillInfraError(msg)

    rule_id = str(rule.get("id", "")).strip()
    rule_file_raw = str(rule.get("file", "")).strip()
    if not rule_file_raw:
        msg = f"ast-grep rule '{rule_id}' is missing 'file'"
        raise SkillUsageError(msg)

    rule_file = Path(rule_file_raw)
    if not rule_file.is_absolute():
        rule_file = (skill_dir / rule_file_raw).resolve()
    if not rule_file.exists():
        msg = f"ast-grep rule file not found: {rule_file}"
        raise SkillUsageError(msg)

    group = str(rule.get("group", rule_id)).strip() or rule_id
    count_by = str(rule.get("count_by", "aggregate")).strip() or "aggregate"
    if count_by not in {"aggregate", "rule_id"}:
        msg = f"Invalid count_by for rule '{rule_id}': {count_by}"
        raise SkillUsageError(msg)
    match_mode = str(rule.get("match", "present")).strip() or "present"
    if match_mode not in {"present", "absent"}:
        msg = f"Invalid match mode for '{rule_id}': {match_mode}"
        raise SkillUsageError(msg)

    command = ["sg", "scan", "--rule", str(rule_file), "--json=stream"]
    for pattern in include_globs:
        command.extend(["--globs", pattern])
    for pattern in exclude_globs:
        command.extend(["--globs", f"!{pattern}"])
    command.append(str(project_path))

    result = run_cmd(command, cwd=project_path, timeout=300)
    if result.returncode not in {0, 1}:
        stderr = (result.stderr or "").strip()
        msg = f"ast-grep failed for {project_name}/{rule_id}: {stderr}"
        raise SkillInfraError(msg)

    grouped: dict[str, int] = {}
    raw_matches = 0

    for raw_line in (result.stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue

        file_value = str(payload.get("file", "")).strip()
        if file_value:
            file_path = Path(file_value)
            if file_path.is_absolute():
                try:
                    rel = normalize_rel_path(str(file_path.relative_to(project_path)))
                except ValueError:
                    rel = normalize_rel_path(file_path.name)
            else:
                rel = normalize_rel_path(file_value)
            if allowed_files and rel and rel not in allowed_files:
                continue

        raw_matches += 1
        key = str(payload.get("ruleId", "unknown")) if count_by == "rule_id" else group
        grouped[key] = grouped.get(key, 0) + 1

    if match_mode == "absent":
        if raw_matches == 0:
            return ({group: 1}, 0)
        return ({}, raw_matches)

    return grouped, raw_matches


def build_custom_command(script: Path) -> list[str]:
    """Build command argv for a custom validator script."""
    if script.suffix == ".py":
        return [sys.executable, str(script)]
    return [str(script)]


def run_custom_rule(
    *,
    rule: dict[str, object],
    skill_dir: Path,
    project_path: Path,
    mode: str,
) -> int:
    """Run one custom rule script and return violation count."""
    rule_id = str(rule.get("id", "")).strip()
    script_raw = str(rule.get("script", "")).strip()
    if not script_raw:
        msg = f"custom rule '{rule_id}' is missing 'script'"
        raise SkillUsageError(msg)

    script = Path(script_raw)
    if not script.is_absolute():
        script = (skill_dir / script_raw).resolve()
    if not script.exists():
        msg = f"custom script not found: {script}"
        raise SkillUsageError(msg)

    command = build_custom_command(script)
    command.extend(["--root", str(project_path)])
    args_obj = rule.get("args", [])
    if args_obj is not None:
        if not isinstance(args_obj, list) or not all(
            isinstance(item, str) for item in args_obj
        ):
            msg = f"custom rule '{rule_id}' has invalid 'args' (expected list[str])"
            raise SkillUsageError(msg)
        command.extend([str(item) for item in args_obj])
    if bool(rule.get("pass_mode")):
        command.extend(["--mode", mode])

    result = run_cmd(command, cwd=project_path, timeout=300)
    if result.returncode == 2:
        stderr = (result.stderr or "").strip()
        msg = f"custom rule invalid args for '{rule_id}': {stderr}"
        raise SkillUsageError(msg)
    if result.returncode not in {0, 1}:
        stderr = (result.stderr or "").strip()
        msg = f"custom rule runtime failed for '{rule_id}': {stderr}"
        raise SkillInfraError(msg)

    count = parse_count_from_json_lines(result.stdout or "")
    if result.returncode == 1:
        count = max(count, 1)
    return count


def run_custom_validate(
    *,
    custom_script_raw: str,
    skill_dir: Path,
    project_name: str,
    project_path: Path,
    mode: str,
) -> int:
    """Run custom skill-level validation script for one project."""
    script = Path(custom_script_raw)
    if not script.is_absolute():
        script = (skill_dir / custom_script_raw).resolve()
    if not script.exists():
        msg = f"custom_validate script not found: {script}"
        raise SkillUsageError(msg)

    command = build_custom_command(script)
    command.extend(["--root", str(project_path), "--mode", mode])
    result = run_cmd(command, cwd=project_path, timeout=300)

    if result.returncode == 2:
        msg = f"custom_validate invalid args on project {project_name}"
        raise SkillUsageError(msg)
    if result.returncode not in {0, 1}:
        stderr = (result.stderr or "").strip()
        msg = f"custom_validate failed on {project_name}: {stderr}"
        raise SkillInfraError(msg)

    count = parse_count_from_json_lines(result.stdout or "")
    if result.returncode == 1:
        count = max(count, 1)
    return count


def compare_baseline(
    *,
    current_counts: dict[str, int],
    baseline_counts: dict[str, int],
    strategy: str,
) -> tuple[bool, dict[str, int], int, int]:
    """Compare current violations against stored baseline counts."""
    if strategy not in {"total", "per_group"}:
        msg = f"Unsupported baseline strategy: {strategy}"
        raise SkillUsageError(msg)

    all_groups = sorted(set(current_counts) | set(baseline_counts))
    deltas: dict[str, int] = {}
    for group in all_groups:
        deltas[group] = current_counts.get(group, 0) - baseline_counts.get(group, 0)

    current_total = sum(current_counts.values())
    baseline_total = sum(baseline_counts.values())

    if strategy == "total":
        passed = current_total <= baseline_total
    else:
        passed = all(delta <= 0 for delta in deltas.values())

    return passed, deltas, current_total, baseline_total


def _extract_fixes(rules_list: list[object]) -> list[dict[str, object]]:
    """Extract fix metadata from each rule entry.

    Strict flat-key format only.  Each rule carries fix metadata via
    top-level keys: ``fix_auto``, ``fix_type``, ``fix_instruction``,
    ``fix_file``, ``fix_description``.

    A nested ``fix:`` sub-dict is rejected as a schema error so that
    rules.yml files stay uniform.
    """
    fixes: list[dict[str, object]] = []
    for rule in rules_list:
        if not isinstance(rule, dict):
            continue
        rule_id = str(rule.get("id", "unknown"))

        # Reject nested fix: — flat keys are the only supported format
        if isinstance(rule.get("fix"), dict):
            msg = (
                f"Rule '{rule_id}' uses nested 'fix:' sub-dict. "
                "Use flat keys instead: fix_auto, fix_type, fix_instruction, "
                "fix_file, fix_description."
            )
            raise SkillUsageError(msg)

        has_fix = any(
            key.startswith("fix_") and rule.get(key) is not None
            for key in rule
            if isinstance(key, str)
        )
        if has_fix:
            entry: dict[str, object] = {"rule_id": rule_id}
            if rule.get("fix_auto") is not None:
                entry["auto"] = rule["fix_auto"]
            if rule.get("fix_type") is not None:
                entry["type"] = rule["fix_type"]
            if rule.get("fix_instruction") is not None:
                entry["instruction"] = rule["fix_instruction"]
            if rule.get("fix_file") is not None:
                entry["file"] = rule["fix_file"]
            if rule.get("fix_description") is not None:
                entry["description"] = rule["fix_description"]
            fixes.append(entry)

    return fixes


def discover_skills(skills_dir: Path) -> list[tuple[str, Path]]:
    """Discover skills that define a rules.yml file."""
    if not skills_dir.exists():
        return []
    found: list[tuple[str, Path]] = []
    for child in sorted(skills_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        rules = child / "rules.yml"
        if rules.exists():
            found.append((child.name, rules))
    return found


def normalize_string_list(value: object, field_name: str) -> list[str]:
    """Validate and normalize a list[str] configuration field."""
    if value is None:
        return []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if not isinstance(item, str):
                msg = f"{field_name} must be a list[str]"
                raise SkillUsageError(msg)
            out.append(item)
        return out
    msg = f"{field_name} must be a list[str]"
    raise SkillUsageError(msg)


def resolve_scan_projects(
    *,
    scan_projects: object,
    discovered: dict[str, object],
    cli_projects: list[str],
) -> list[str]:
    """Resolve selected scan targets from config and CLI filters."""
    auto = ["."]
    flext_values = discovered.get("flext", [])
    external_values = discovered.get("external", [])
    if isinstance(flext_values, list):
        auto.extend([x for x in flext_values if isinstance(x, str)])
    if isinstance(external_values, list):
        auto.extend([x for x in external_values if isinstance(x, str)])
    auto = unique_sorted(auto)

    if scan_projects is None:
        selected = auto
    elif isinstance(scan_projects, list):
        scan_list: list[str] = []
        for item in scan_projects:
            if not isinstance(item, str):
                msg = "scan_targets.projects must be a list[str]"
                raise SkillUsageError(msg)
            scan_list.append(item)
        if not scan_list or scan_list == ["auto"]:
            selected = auto
        else:
            selected = unique_sorted(scan_list)
    else:
        msg = "scan_targets.projects must be null or list[str]"
        raise SkillUsageError(msg)

    if cli_projects:
        allowed = set(cli_projects)
        selected = [name for name in selected if name in allowed]

    return selected


def validate_skill(
    *,
    root: Path,
    mode: str,
    skill_name: str,
    rules_path: Path,
    update_baseline: bool,
    cli_projects: list[str],
    discovered_projects: dict[str, object],
    project_lookup: dict[str, Path],
) -> tuple[bool, dict[str, object]]:
    """Validate one skill across selected projects."""
    rules = load_rules_yml(rules_path)
    skill_dir = rules_path.parent.resolve()

    scan_targets_obj = rules.get("scan_targets", {})
    if scan_targets_obj is None:
        scan_targets_obj = {}
    if not isinstance(scan_targets_obj, dict):
        msg = f"scan_targets must be a mapping: {rules_path}"
        raise SkillUsageError(msg)

    include_globs = normalize_string_list(
        scan_targets_obj.get("include", ["**/*.py"]),
        "scan_targets.include",
    )
    if not include_globs:
        include_globs = ["**/*"]
    exclude_globs = normalize_string_list(
        scan_targets_obj.get("exclude", []),
        "scan_targets.exclude",
    )
    all_known: set[str] = set()
    for group in discovered_projects.values():
        if isinstance(group, list):
            all_known.update(str(p) for p in group)
    exclude_globs = normalize_exclude_globs(exclude_globs, all_known)

    selected_projects = resolve_scan_projects(
        scan_projects=scan_targets_obj.get("projects", []),
        discovered=discovered_projects,
        cli_projects=cli_projects,
    )
    if not selected_projects:
        msg = f"No projects selected for skill '{skill_name}'"
        raise SkillUsageError(msg)
    for name in selected_projects:
        if name not in project_lookup:
            msg = f"Unknown project '{name}' in skill '{skill_name}'"
            raise SkillUsageError(msg)

    baseline_obj = rules.get("baseline", {})
    report_obj = rules.get("report", {})
    if baseline_obj is None:
        baseline_obj = {}
    if report_obj is None:
        report_obj = {}
    if not isinstance(baseline_obj, dict) or not isinstance(report_obj, dict):
        msg = "baseline/report must be mappings"
        raise SkillUsageError(msg)

    baseline_strategy = str(baseline_obj.get("strategy", "total"))
    baseline_path = render_path_template(
        root,
        str(baseline_obj.get("file", BASELINE_DEFAULT)),
        skill_name,
        BASELINE_DEFAULT,
    )
    report_path = render_path_template(
        root,
        str(report_obj.get("file", REPORT_DEFAULT)),
        skill_name,
        REPORT_DEFAULT,
    )

    rules_obj = rules.get("rules", [])
    if rules_obj is None:
        rules_obj = []
    if not isinstance(rules_obj, list):
        msg = "rules must be a list"
        raise SkillUsageError(msg)

    valid_fix_types = {"ast-grep", "custom"}
    for rule in rules_obj:
        if not isinstance(rule, dict):
            continue
        rid = rule.get("id", "?")
        ft = rule.get("fix_type")
        fa = rule.get("fix_auto", False)
        if ft == "manual":
            eprint(
                f"  ERROR [{rid}]: fix_type: manual is invalid. Remove fix_type when fix_auto: false."
            )
        if ft and ft not in valid_fix_types and ft != "manual":
            eprint(
                f"  ERROR [{rid}]: fix_type: '{ft}' is invalid. Valid: {sorted(valid_fix_types)}"
            )
        if fa and not ft:
            fix_file = rule.get("fix_file")
            if not fix_file:
                msg = (
                    f"[{rid}] fix_auto=true requires fix_type/fix_file; "
                    "auto-fix cannot be skipped in strict mode"
                )
                raise SkillUsageError(msg)

    counts: dict[str, int] = {}
    per_project_counts: dict[str, dict[str, int]] = {}

    print(f"\n{'=' * 72}")
    print(f"Skill: {skill_name}")
    print(f"Mode: {mode}")
    print(f"Projects: {', '.join(selected_projects)}")
    print(f"{'=' * 72}")

    for project_name in selected_projects:
        project_path = project_lookup[project_name].resolve()
        tracked = get_tracked_files(project_path)
        selected_files = filter_files(
            tracked, include_globs, exclude_globs, project_path
        )
        allowed_files = set(selected_files)

        print(f"\n  Project: {project_name}")
        print(f"    tracked={len(tracked)} selected={len(selected_files)}")

        project_groups: dict[str, int] = {}

        for rule_obj in rules_obj:
            if not isinstance(rule_obj, dict):
                msg = "Each rule must be a mapping"
                raise SkillUsageError(msg)

            rule_id = str(rule_obj.get("id", "")).strip()
            if not rule_id:
                msg = "Rule missing id"
                raise SkillUsageError(msg)

            rule_type = str(rule_obj.get("type", "")).strip()
            group = str(rule_obj.get("group", rule_id)).strip() or rule_id

            if rule_type == "ast-grep":
                grouped, raw = run_ast_grep_rule(
                    rule=rule_obj,
                    skill_dir=skill_dir,
                    project_name=project_name,
                    project_path=project_path,
                    include_globs=include_globs,
                    exclude_globs=exclude_globs,
                    allowed_files=allowed_files,
                )
                for key, amount in grouped.items():
                    counts[key] = counts.get(key, 0) + amount
                    project_groups[key] = project_groups.get(key, 0) + amount
                print(f"    [{rule_id}] ast-grep matches={raw}")

            elif rule_type == "custom":
                custom_count = run_custom_rule(
                    rule=rule_obj,
                    skill_dir=skill_dir,
                    project_path=project_path,
                    mode=mode,
                )
                counts[group] = counts.get(group, 0) + custom_count
                project_groups[group] = project_groups.get(group, 0) + custom_count
                print(f"    [{rule_id}] custom violations={custom_count}")

            else:
                msg = f"Unsupported rule type '{rule_type}' in rule '{rule_id}'"
                raise SkillUsageError(msg)

        custom_validate = rules.get("custom_validate")
        if isinstance(custom_validate, str) and custom_validate.strip():
            cv_count = run_custom_validate(
                custom_script_raw=custom_validate.strip(),
                skill_dir=skill_dir,
                project_name=project_name,
                project_path=project_path,
                mode=mode,
            )
            counts["custom_validate"] = counts.get("custom_validate", 0) + cv_count
            project_groups["custom_validate"] = (
                project_groups.get("custom_validate", 0) + cv_count
            )
            print(f"    [custom_validate] violations={cv_count}")

        per_project_counts[project_name] = dict(sorted(project_groups.items()))

    counts = dict(sorted(counts.items()))
    total = sum(counts.values())

    baseline_exists = baseline_path.exists()
    baseline_data = read_json(baseline_path) if baseline_exists else {}

    if update_baseline:
        baseline_payload: dict[str, object] = {
            "skill": skill_name,
            "strategy": baseline_strategy,
            "counts": counts,
            "total": total,
            "updated_at": int(time.time()),
        }
        write_json(baseline_path, baseline_payload)
        baseline_data = baseline_payload
        baseline_exists = True
        print(f"\n  Baseline updated: {baseline_path}")

    baseline_initialized = False
    if mode == "baseline" and not baseline_exists:
        baseline_payload = {
            "skill": skill_name,
            "strategy": baseline_strategy,
            "counts": counts,
            "total": total,
            "initialized_at": int(time.time()),
        }
        write_json(baseline_path, baseline_payload)
        baseline_data = baseline_payload
        baseline_exists = True
        baseline_initialized = True
        print(f"\n  Baseline initialized: {baseline_path}")

    raw_baseline_counts = baseline_data.get("counts", {}) if baseline_data else {}
    if raw_baseline_counts and not isinstance(raw_baseline_counts, dict):
        msg = f"Invalid baseline counts format: {baseline_path}"
        raise SkillUsageError(msg)

    baseline_counts: dict[str, int] = {}
    if isinstance(raw_baseline_counts, dict):
        for key, value in raw_baseline_counts.items():
            if isinstance(value, int):
                baseline_counts[str(key)] = value

    comparison: dict[str, object] = {
        "strategy": baseline_strategy,
        "passed": True,
        "deltas": {},
        "current_total": total,
        "baseline_total": 0,
        "initialized": baseline_initialized,
    }

    if mode == "strict":
        passed = total == 0
        comparison["passed"] = passed
        comparison["deltas"] = {"total": total}
        comparison["baseline_total"] = 0
    elif baseline_initialized:
        passed = True
        comparison["deltas"] = {"total": 0}
        comparison["baseline_total"] = total
    else:
        passed, deltas, current_total, baseline_total = compare_baseline(
            current_counts=counts,
            baseline_counts=baseline_counts,
            strategy=baseline_strategy,
        )
        comparison["passed"] = passed
        comparison["deltas"] = deltas
        comparison["current_total"] = current_total
        comparison["baseline_total"] = baseline_total

    report: dict[str, object] = {
        "skill": skill_name,
        "mode": mode,
        "projects_scanned": selected_projects,
        "scan_succeeded": True,
        "counts": counts,
        "total": total,
        "per_project": per_project_counts,
        "baseline_comparison": comparison,
        "fix_summary": _extract_fixes(rules_obj),
    }

    write_json(report_path, report)
    print(f"\n  Report: {report_path}")
    print(f"  Total violations: {total}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return passed, report


def list_skills(skills_dir: Path) -> int:
    """List skills and their configured rule metadata."""
    skills = discover_skills(skills_dir)
    if not skills:
        print("No skills with rules.yml found")
        return EXIT_PASS

    print(f"{'Skill':<40} {'Rules':<7} {'Custom':<7}")
    print("-" * 62)
    for skill_name, rules_path in skills:
        try:
            rules = load_rules_yml(rules_path)
            rules_obj = rules.get("rules", [])
            count = len(rules_obj) if isinstance(rules_obj, list) else 0
            has_custom = "yes" if rules.get("custom_validate") else "no"
            print(f"{skill_name:<40} {count:<7} {has_custom:<7}")
        except (SkillUsageError, SkillInfraError) as exc:
            print(f"{skill_name:<40} {'ERR':<7} {'ERR':<7} {exc}")
    return EXIT_PASS


def list_projects(root: Path) -> int:
    """List discovered workspace projects."""
    discovered = discover_projects(root)
    raw_flext = discovered.get("flext", [])
    raw_external = discovered.get("external", [])
    flext = (
        [x for x in raw_flext if isinstance(x, str)]
        if isinstance(raw_flext, list)
        else []
    )
    external = (
        [x for x in raw_external if isinstance(x, str)]
        if isinstance(raw_external, list)
        else []
    )

    print(f"Root project: {discovered.get('root', '.')}")
    print(f"FLEXT projects ({len(flext)}):")
    for name in flext:
        print(f"  {name}")
    print(f"External projects ({len(external)}):")
    for name in external:
        print(f"  {name}")
    print(f"Total scan targets (including root): {1 + len(flext) + len(external)}")
    return EXIT_PASS


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse validator CLI arguments."""
    parser = argparse.ArgumentParser(description="Generic data-driven skill validator")
    _ = parser.add_argument("--skill", help="Validate one skill by folder name")
    _ = parser.add_argument("--all", action="store_true", help="Validate all skills")
    _ = parser.add_argument("--list-skills", action="store_true", help="List skills")
    _ = parser.add_argument(
        "--list-projects", action="store_true", help="List projects"
    )
    _ = parser.add_argument(
        "--mode", choices=["baseline", "strict"], default="baseline"
    )
    _ = parser.add_argument("--root", default=".", help="Workspace root")
    _ = parser.add_argument("--skills-dir", default=str(SKILLS_DIR), help="Skills root")
    _ = parser.add_argument("--update-baseline", action="store_true")
    _ = parser.add_argument(
        "--project", action="append", default=[], help="Project filter"
    )

    args = parser.parse_args(argv)

    action_count = (
        int(args.list_skills)
        + int(args.list_projects)
        + int(args.all)
        + int(bool(args.skill))
    )
    if action_count == 0:
        msg = "Specify one of: --skill, --all, --list-skills, --list-projects"
        raise SkillUsageError(msg)
    if args.all and args.skill:
        msg = "Use either --all or --skill"
        raise SkillUsageError(msg)
    if (args.list_skills or args.list_projects) and (args.all or args.skill):
        msg = "Listing options cannot be combined with validation options"
        raise SkillUsageError(msg)

    return args


def run_main(argv: list[str]) -> int:
    """Run validator workflow and return standardized exit code."""
    try:
        args = parse_args(argv)
    except SkillUsageError as exc:
        eprint(f"ERROR: {exc}")
        return EXIT_USAGE
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else EXIT_USAGE
        return code if code in {0, 1, 2, 3} else EXIT_USAGE

    root = Path(args.root).resolve()
    skills_dir = Path(args.skills_dir)
    if not skills_dir.is_absolute():
        skills_dir = (root / skills_dir).resolve()

    try:
        if args.list_skills:
            return list_skills(skills_dir)
        if args.list_projects:
            return list_projects(root)
    except SkillUsageError as exc:
        eprint(f"ERROR: {exc}")
        return EXIT_USAGE
    except SkillInfraError as exc:
        eprint(f"ERROR: {exc}")
        return EXIT_INFRA

    try:
        discovered = discover_projects(root)
        lookup = build_project_lookup(root, discovered)
    except SkillUsageError as exc:
        eprint(f"ERROR: {exc}")
        return EXIT_USAGE
    except SkillInfraError as exc:
        eprint(f"ERROR: {exc}")
        return EXIT_INFRA

    cli_projects = [str(name) for name in args.project]
    for project in cli_projects:
        if project not in lookup:
            eprint(f"ERROR: Unknown project passed via --project: {project}")
            return EXIT_USAGE

    if args.all:
        skills = discover_skills(skills_dir)
        if not skills:
            print("No skills with rules.yml found")
            return EXIT_PASS
    else:
        skill_name = str(args.skill)
        rules_path = skills_dir / skill_name / "rules.yml"
        if not rules_path.exists():
            eprint(f"ERROR: rules.yml not found for skill '{skill_name}'")
            return EXIT_USAGE
        skills = [(skill_name, rules_path)]

    reports: list[dict[str, object]] = []
    pass_map: dict[str, bool] = {}
    failed_skills: list[str] = []
    usage_errors: list[str] = []
    infra_errors: list[str] = []

    for skill_name, rules_path in skills:
        try:
            passed, report = validate_skill(
                root=root,
                mode=str(args.mode),
                skill_name=skill_name,
                rules_path=rules_path,
                update_baseline=bool(args.update_baseline),
                cli_projects=cli_projects,
                discovered_projects=discovered,
                project_lookup=lookup,
            )
            reports.append(report)
            pass_map[skill_name] = passed
            if not passed:
                failed_skills.append(skill_name)
        except SkillUsageError as exc:
            pass_map[skill_name] = False
            usage_errors.append(f"{skill_name}: {exc}")
        except SkillInfraError as exc:
            pass_map[skill_name] = False
            infra_errors.append(f"{skill_name}: {exc}")

    print(f"\n{'=' * 72}")
    print(f"Validation summary: {len(skills)} skill(s)")
    for skill_name, _rules_path in skills:
        status = "PASS" if pass_map.get(skill_name) else "FAIL"
        total = 0
        for report in reports:
            if report.get("skill") == skill_name:
                maybe_total = report.get("total", 0)
                if isinstance(maybe_total, int):
                    total = maybe_total
                break
        print(f"  {skill_name}: {status} total={total}")

    if failed_skills:
        print(f"Failed skills: {', '.join(failed_skills)}")
    if usage_errors:
        print("Usage/config errors:")
        for item in usage_errors:
            print(f"  {item}")
    if infra_errors:
        print("Infra/runtime errors:")
        for item in infra_errors:
            print(f"  {item}")

    overall_pass = not failed_skills and not usage_errors and not infra_errors
    print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
    print(f"{'=' * 72}")

    if infra_errors:
        return EXIT_INFRA
    if usage_errors:
        return EXIT_USAGE
    if failed_skills:
        return EXIT_FAIL
    return EXIT_PASS


def main() -> None:
    """Execute CLI entry point and exit with normalized code."""
    code = run_main(sys.argv[1:])
    if code not in {EXIT_PASS, EXIT_FAIL, EXIT_USAGE, EXIT_INFRA}:
        code = EXIT_INFRA
    raise SystemExit(code)


if __name__ == "__main__":
    main()
