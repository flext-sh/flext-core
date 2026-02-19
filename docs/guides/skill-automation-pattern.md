<!-- Generated from docs/guides/skill-automation-pattern.md for flext-core. -->
<!-- Source of truth: workspace docs/guides/. -->

# flext-core - Skill Automation Pattern

> Project profile: `flext-core`



<!-- TOC START -->
- [Goal](#goal)
- [Required Outputs](#required-outputs)
- [Standard Skill Contract](#standard-skill-contract)
- [Standard Skill Format](#standard-skill-format)
- [Implementation Checklist](#implementation-checklist)
- [Example (Current Pattern)](#example-current-pattern)
- [Verification Commands](#verification-commands)
- [Adoption Rule](#adoption-rule)
<!-- TOC END -->

This guide defines the standard way to create reusable automation skills in this repository.

## Goal

Create automations that are reproducible, script-first, and enforceable by CI-style commands.

## Required Outputs

For each new automation family, deliver all items below:

1. One skill folder: `.claude/skills/<automation-name>/` containing:
   - `SKILL.md` — canonical skill document
   - `rules.yml` — detection rules (ast-grep, ripgrep, or custom)
   - `rules/` — ast-grep rule files (if any)
   - `baseline.json` — violation baseline (auto-generated)
2. One docs page in `docs/guides/` (if cross-cutting)

## Standard Skill Contract

Skills are validated by the generic runner:

```bash
python3 scripts/core/skill_validate.py --skill <name>
python3 scripts/core/skill_validate.py --skill <name> --mode strict
python3 scripts/core/skill_validate.py --skill <name> --update-baseline
```

The runner auto-discovers all skills:

```bash
python3 scripts/core/skill_validate.py --all
```

## Standard Skill Format

The skill must follow the canonical format from `skill-format-universal` and include:

- Concrete paths under `## Scope`
- Existing anchors under `## References`
- Enforceable behaviors under `## Rules`
- Copyable commands under `## Instructions`
- Ordered execution in `## Workflow`
- Good/Bad examples under `## Examples`
- Executable checks under `## Verification`

## Implementation Checklist

1. Define the invariant (policy or quality requirement).
2. Create `rules.yml` with detection rules (ast-grep, ripgrep, or custom).
3. Place ast-grep rule files in skill `rules/` directory.
4. Initialize baseline with `python3 scripts/core/skill_validate.py --skill <name> --update-baseline`.
5. Write or update skill doc with exact commands.
6. Add or update a docs guide in `docs/guides/` (if cross-cutting).
7. Run `python3 scripts/core/skill_validate.py --all` to verify integration.

## Example (Current Pattern)

Current repository implementation uses the **self-contained skill architecture**. Each skill
folder (`.claude/skills/<skill>/`) owns its own `rules.yml`, `rules/` ast-grep files,
`baseline.json`, and `report.json`. The generic runner `scripts/core/skill_validate.py`
discovers and executes everything.

**Dict/Any Policy Gate**:
- Skill: `.claude/skills/flext-strict-typing/SKILL.md`
- Rules: `.claude/skills/flext-strict-typing/rules.yml` (10 rules: 8 ast-grep + 2 ripgrep)
- AST rules: `.claude/skills/flext-strict-typing/rules/*.yml`
- Baseline: `.claude/skills/flext-strict-typing/baseline.json`

**Pydantic v2 Policy Gate**:
- Skill: `.claude/skills/lib-pydantic-v2/SKILL.md`
- Rules: `.claude/skills/lib-pydantic-v2/rules.yml` (8 ast-grep rules)
- AST rules: `.claude/skills/lib-pydantic-v2/rules/*.yml`
- Baseline: `.claude/skills/lib-pydantic-v2/baseline.json`

**Generic runner**:
- `scripts/core/skill_validate.py` — auto-discovers `.claude/skills/*/rules.yml`

## Verification Commands

```bash
python3 scripts/core/skill_validate.py --list-skills
python3 scripts/core/skill_validate.py --skill flext-strict-typing
python3 scripts/core/skill_validate.py --skill lib-pydantic-v2
python3 scripts/core/skill_validate.py --all
```

## Adoption Rule

For future automation work, do not introduce manual-only procedures. Ship scripts + skill + docs together in the same change.
