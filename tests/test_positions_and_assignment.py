import random

from conftest import make_player, make_squad

from assignment import INCOMPATIBLE, assign_slots, hungarian_max
from positions import parse_position_string, parse_slot_position, player_can_play_slot, position_familiarity
from profiles import PROFILES
from scorer import get_best_11


def test_parse_position_string_expands_sides():
    got = dict(parse_position_string("D (RC), AM (RL), ST (C)"))
    assert set(got["D"]) == {"R", "C"}
    assert set(got["AM"]) == {"R", "L"}
    assert got["ST"] == "C"


def test_parse_slot_position_is_side_aware():
    assert parse_slot_position("AML") == ("AM", "L")
    assert parse_slot_position("AMR") == ("AM", "R")
    assert parse_slot_position("DCR") == ("D", "C")
    assert parse_slot_position("DMCL") == ("DM", "C")
    assert parse_slot_position("GK") == ("GK", "")


def test_right_winger_is_not_natural_on_the_left():
    right = position_familiarity("AM (R)", "ifa", "AML")
    left = position_familiarity("AM (L)", "ifa", "AML")
    assert left == 1.0
    assert 0 < right < 1.0  # wrong flank is a penalty, not a hard block by default
    assert player_can_play_slot("AM (R)", "AML", "ifa", wrong_flank=0.0) is False


def test_centre_back_cannot_play_wing():
    assert position_familiarity("D (C)", "ifa", "AML") == 0.0
    assert position_familiarity("D (RC)", "wbs", "DR") == 1.0


def test_hungarian_matches_brute_force():
    from itertools import permutations

    rng = random.Random(7)
    for _ in range(200):
        n = rng.randint(1, 5)
        m = rng.randint(n, 6)
        mat = [[rng.choice([INCOMPATIBLE, rng.uniform(0, 20)]) for _ in range(m)] for _ in range(n)]
        got = hungarian_max(mat)
        best = None
        for perm in permutations(range(m), n):
            score = sum(mat[i][j] for i, j in enumerate(perm) if mat[i][j] != INCOMPATIBLE)
            filled = sum(1 for i, j in enumerate(perm) if mat[i][j] != INCOMPATIBLE)
            key = (filled, score)
            if best is None or key > best:
                best = key
        got_filled = sum(1 for i, j in enumerate(got) if j is not None and mat[i][j] != INCOMPATIBLE)
        got_score = sum(mat[i][j] for i, j in enumerate(got) if j is not None and mat[i][j] != INCOMPATIBLE)
        assert (got_filled, round(got_score, 6)) == (best[0], round(best[1], 6))


def test_assign_slots_leaves_unfillable_slot_empty():
    res = assign_slots(["a", "b"], ["p1"], lambda s, p: 5.0 if s == "a" else INCOMPATIBLE)
    assert res["a"] == ("p1", 5.0)
    assert res["b"][0] is None


def test_best_11_does_not_let_versatile_player_block_specialist():
    """Versatile player is slightly better everywhere but the global optimum
    puts them where the specialist can't go."""
    formation = [{"pos": "GK", "role": "sks"}, {"pos": "AMR", "role": "ifa"}, {"pos": "AML", "role": "ifa"}]
    rows = [
        make_player("Keeper", "GK", attrs={"Ref": 15, "Han": 15}),
        make_player("Versatile", "AM (RL)", attrs={"Pac": 16, "Acc": 16, "Dri": 16, "Fin": 16}),
        make_player("LeftOnly", "AM (L)", attrs={"Pac": 15, "Acc": 15, "Dri": 15, "Fin": 15}),
    ]
    df = make_squad(rows)
    xi = get_best_11(df, formation, PROFILES["default"])
    by_pos = {s["pos"]: s["player_name"] for s in xi}
    assert by_pos["GK"] == "Keeper"
    assert by_pos["AML"] == "LeftOnly"
    assert by_pos["AMR"] == "Versatile"


def test_best_11_keeps_slot_unfilled_when_no_compatible_player():
    formation = [{"pos": "GK", "role": "sks"}, {"pos": "STC", "role": "afa"}]
    df = make_squad([make_player("Keeper", "GK")])
    xi = get_best_11(df, formation, PROFILES["default"])
    st = next(s for s in xi if s["pos"] == "STC")
    assert st["player_idx"] == -1
    assert st["score"] == 0
