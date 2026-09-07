---
applyTo: '**/*.py,**/*.md,**/*.toml,**/*.yaml,**/*.yml'
---

# Coordinate shared-file edits; never clobber WIP

Multiple agents may edit the same files concurrently. Declare file ownership in
the active collaboration channel before editing; do not add coordination-only
comments to product code or documentation.

The complete preservation, fix-forward, and severe-conflict contract is owned by
`fix-forward collaboration` (rule file).

- Re-read a mutable file right before editing; converge, never revert another
  actor's valid change.
- Never overwrite or stash uncommitted WIP. Preserve durable evidence in the next
  canonical commit/PR/CI artifact.
- The checkout's current branch is shared state too. When another actor may hold
  the same working tree, re-read the branch immediately before committing and
  verify the commit landed on the intended ref: a concurrent switch relocates
  your commit onto their branch with no conflict and no warning. Recover by
  attributable cherry-pick onto the intended branch and tell the other actor;
  never rewrite either branch to remove the misplaced commit.
- An uncommitted edit in a shared working tree is not private state. Another
  actor's checkout can discard it with no conflict, no warning, and no trace in
  any log — the file simply reads as it did before. Commit at every material
  checkpoint rather than holding work in the tree, and when an edit is gone,
  re-preflight and reapply: absence proves no actor and no intent.
- Resolve the tracker mode from the repository's own instructions before any
  tracker action; while that authority declares suspension, invoke no tracker
  command, create no substitute tracker or ledger, and keep phase closure
  unavailable.
