"""Repository DAG construction and topological sorting.

Implements ``BuildRepoDAG(F)`` from Eq. 1 of the NERFIFY paper:

    C = (F, G)
    F = {f1, f2, …, fn}
    G = BuildRepoDAG(F)          # directed acyclic graph
    V(G) = F                     # vertices = files
    E(G) ⊆ F × F                 # edges = import / dataflow deps
    Acyclicity: (fi, fj) ∈ E(G) ⟹ no path from fj to fi

Uses Kahn's algorithm for O(V+E) topological sort with explicit
cycle detection.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Sequence, Tuple


class CyclicDependencyError(Exception):
    """Raised when the file dependency graph contains a cycle."""

    def __init__(self, cycle_members: Sequence[str]) -> None:
        self.cycle_members = list(cycle_members)
        super().__init__(
            f"Circular dependency detected among: {', '.join(self.cycle_members)}"
        )


def build_repo_dag(
    files: Sequence[str],
    dependencies: Dict[str, List[str]],
) -> Tuple[Dict[str, List[str]], List[str]]:
    """Build a repository DAG and return its topological order.

    Parameters
    ----------
    files : sequence of str
        The set F = {f1, …, fn} of filenames in the repository.
    dependencies : dict
        Mapping ``file → [files it depends on]``.  An edge (u → v)
        means *v depends on u* (i.e. v imports from u).  Every key
        and every value element must be present in *files*.

    Returns
    -------
    adjacency : dict
        Forward adjacency list of the validated DAG.
        ``adjacency[u]`` = list of files that *u* feeds into.
    topo_order : list of str
        Files in topological order (leaves / zero-in-degree first).
        Code generation should proceed in this order.

    Raises
    ------
    CyclicDependencyError
        If the dependency graph contains one or more cycles.
    ValueError
        If a dependency references a file not in *files*.
    """

    file_set = set(files)

    # ── Validate references ─────────────────────────────────────────────
    for fname, deps in dependencies.items():
        if fname not in file_set:
            raise ValueError(
                f"File '{fname}' appears in dependencies but not in the file set."
            )
        for dep in deps:
            if dep not in file_set:
                raise ValueError(
                    f"Dependency '{dep}' (required by '{fname}') is not in the file set."
                )

    # ── Build adjacency + in-degree ─────────────────────────────────────
    # Edge semantics: dep → fname  (dep is upstream of fname)
    adjacency: Dict[str, List[str]] = {f: [] for f in files}
    in_degree: Dict[str, int] = {f: 0 for f in files}

    for fname, deps in dependencies.items():
        for dep in deps:
            adjacency[dep].append(fname)
            in_degree[fname] += 1

    # ── Kahn's algorithm ────────────────────────────────────────────────
    queue: deque[str] = deque()
    for f in files:
        if in_degree[f] == 0:
            queue.append(f)

    topo_order: List[str] = []

    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for neighbour in adjacency[node]:
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    # ── Cycle detection ─────────────────────────────────────────────────
    if len(topo_order) != len(files):
        remaining = [f for f in files if f not in set(topo_order)]
        raise CyclicDependencyError(remaining)

    return adjacency, topo_order
