# Adopt the current worktree and never discard it with Git

Two `git reset` runs wiped multi-agent worktrees and endangered shared state.
Treat every current change in an authorized repository or worktree as owned input,
regardless of when it appeared or who authored it. These are forbidden as change
management operations:

`git reset`, `git checkout -- .`, `git restore`, `git clean -xdf`/`-Xdf`,
`git stash`, `git rebase`, `git revert`, `git push --force`.

- Stage only reviewed, intentional paths (`git add <scoped paths>`); never
  `git add -A`/`.` at a workspace or umbrella root.
- Apply `rules/coordination/fix-forward-collaboration.md`; recover evidence from
  `git reflog` only when authorized, never by replacing the adopted current state.
- Commit often so the combined work survives a lane or process failure.

See also: `operator-precedence.md` (rule file) — integration authority.
