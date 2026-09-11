"""Rope runtime patches (private).

Holds intentional monkey-patches of rope internals. Each module here patches a
protected rope surface that exposes no public registration API; the pyright
``reportPrivateUsage`` diagnostic is scoped off for this subpackage via a
dedicated ``executionEnvironments`` entry in the root ``pyproject.toml``
(rationale recorded inline there).
"""

__all__: list[str] = []
