---
name: workspace-toolchain
description: 'tool layers, binary owners, workspace generators, host boundaries'
license: MIT
metadata:
  aihub.tags: '["policy:atomic-effects","policy:causal-subprocess","policy:fail-loud","policy:no-fallback","policy:preflight-before-effects","policy:strict-execution","policy:zero-residue","provenance:agents-owned","role:configuration","updates:manual","usage:router"]'
  version: 1.0.0
---

# Workspace Toolchain

Activate when a task needs a binary, runtime dependency, PATH surface, tool
pin, local environment, or a decision about where that tool belongs. General
code changes without a tool-ownership question do not activate it.

Resolve the tool layer and its one canonical owner before any effect. Host
tools, project fleet tools, project-specific tools, and development
dependencies have different owners and generators; a host decision never
becomes a project pin by proximity or convenience. Follow the
`ownership procedure` (skill file) for the exact routing and
runtime proof.

Never write a generated tool surface by hand or add an undeclared PATH entry,
alias, checkout dependency, keyring, retry, fallback, or duplicate installer.
The first missing owner, conflicting owner, invalid pin, failed generator, or
failed runtime check stops the workflow unchanged.
