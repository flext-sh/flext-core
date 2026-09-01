---
applyTo: '**'
---

# An approved plan owns its topic

At plan start or update, reconcile every correlated owner, WIP, branch, commit,
and PR within the authorized repository. Preserve and adopt useful work into
the existing change branch under
`fix-forward collaboration` (rule file). Destroy, stash, or
revert nothing.

When required work has not reached the integration branch, adopt it into the
owned branch by reviewed non-FF merge or attributable cherry-pick. Preserve
attribution and revalidate the integrated result.

Do not expand to another repository, create a workspace, or invoke a suspended
orchestration/tracker runtime. During suspension, create no substitute tracker
or ledger and preserve evidence only in separately authorized Git/PR/CI.
