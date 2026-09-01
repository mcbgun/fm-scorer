"""Globally optimal player-to-slot assignment (Hungarian / Kuhn-Munkres).

Every part of the app that needs "who starts where" goes through
:func:`assign_slots` so that a versatile player is never greedily locked into a
slot where a specialist would have produced a better *total* line-up.
"""

from collections.abc import Hashable, Sequence

INCOMPATIBLE = float("-inf")


def hungarian_max(matrix: list[list[float]]) -> list[int]:
    """Maximum-weight assignment for a rectangular matrix (rows <= cols).

    Returns ``assign[row] = col``. Cells equal to ``INCOMPATIBLE`` are never
    chosen unless no compatible option exists, in which case the caller should
    treat the pairing as "unfilled".
    """
    n = len(matrix)
    if n == 0:
        return []
    m = len(matrix[0])
    if n > m:
        raise ValueError("hungarian_max requires rows <= cols")

    # Convert to a minimisation problem with finite costs.
    finite = [v for row in matrix for v in row if v != INCOMPATIBLE]
    big = (max(finite) - min(finite) + 1.0) * (n + 1) if finite else 1.0
    top = max(finite) if finite else 0.0
    cost = [[(top - v) if v != INCOMPATIBLE else (top + big) for v in row] for row in matrix]

    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)  # p[col] = row assigned (1-based), 0 = none
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assign = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            assign[p[j] - 1] = j - 1
    return assign


def assign_slots(
    slot_ids: Sequence[Hashable],
    player_ids: Sequence[Hashable],
    score_fn,
) -> dict[Hashable, tuple[Hashable | None, float]]:
    """Assign at most one player per slot maximising total score.

    ``score_fn(slot_id, player_id)`` must return a float, or ``INCOMPATIBLE``
    when the player cannot fill that slot. Returns ``{slot_id: (player_id or
    None, score)}``; slots with no compatible player are returned unfilled.
    """
    slots = list(slot_ids)
    players = list(player_ids)
    result: dict[Hashable, tuple[Hashable | None, float]] = {s: (None, 0.0) for s in slots}
    if not slots or not players:
        return result

    matrix = [[score_fn(s, pl) for pl in players] for s in slots]
    if len(players) < len(slots):
        # Pad with dummy "no one" columns so rows <= cols holds.
        pad = len(slots) - len(players)
        matrix = [row + [INCOMPATIBLE] * pad for row in matrix]

    assign = hungarian_max(matrix)
    for si, col in enumerate(assign):
        if col < 0 or col >= len(players):
            continue
        score = matrix[si][col]
        if score == INCOMPATIBLE:
            continue
        result[slots[si]] = (players[col], score)
    return result
