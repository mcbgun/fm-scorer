"""All 85 FM24 role definitions from squirrel_plays' scoring system.

Each role has three attribute tiers:
  - key:   weight x5 (most important attributes)
  - green: weight x3 (important attributes)
  - blue:  weight x1 (secondary attributes)

Score = (sum(key) * 5 + sum(green) * 3 + sum(blue) * 1) / denominator
where denominator = len(key)*5 + len(green)*3 + len(blue)*1
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RoleDef:
    id: str
    name: str
    key: tuple[str, ...]
    green: tuple[str, ...]
    blue: tuple[str, ...]

    @property
    def denominator(self) -> int:
        return len(self.key) * 5 + len(self.green) * 3 + len(self.blue) * 1


# fmt: off
ROLES: dict[str, RoleDef] = {
    # === Goalkeepers ===
    "gkd":  RoleDef("gkd",  "Goalkeeper - Defend",            ("Agi","Ref"), ("Aer","Cmd","Han","Kic","Cnt","Pos"), ("1v1","Thr","Ant","Dec")),
    "skd":  RoleDef("skd",  "Sweeper Keeper - Defend",        ("Agi","Ref"), ("Cmd","Kic","1v1","Ant","Cnt","Pos"), ("Aer","Fir","Han","Pas","TRO","Dec","Vis","Acc")),
    "sks":  RoleDef("sks",  "Sweeper Keeper - Support",       ("Agi","Ref"), ("Cmd","Kic","1v1","Ant","Cnt","Pos"), ("Aer","Fir","Han","Pas","TRO","Dec","Vis","Acc")),
    "ska":  RoleDef("ska",  "Sweeper Keeper - Attack",        ("Agi","Ref"), ("Cmd","Kic","1v1","Ant","Cnt","Pos"), ("Aer","Fir","Han","Pas","TRO","Dec","Vis","Acc")),

    # === Defenders ===
    "bpdd": RoleDef("bpdd", "Ball Playing Defender - Defend", ("Acc","Pac","Jum","Cmp"), ("Hea","Mar","Pas","Tck","Pos","Str"), ("Fir","Tec","Agg","Ant","Bra","Cnt","Dec","Vis")),
    "bpds": RoleDef("bpds", "Ball Playing Defender - Stopper",("Acc","Pac","Jum","Cmp"), ("Hea","Pas","Tck","Pos","Str","Agg","Bra","Dec"), ("Fir","Tec","Ant","Cnt","Vis","Mar")),
    "bpdc": RoleDef("bpdc", "Ball Playing Defender - Cover",  ("Acc","Pac","Jum","Cmp"), ("Mar","Pas","Tck","Pos","Ant","Cnt","Dec"), ("Fir","Tec","Bra","Vis","Str","Hea")),
    "cdd":  RoleDef("cdd",  "Central Defender - Defend",      ("Acc","Pac","Jum","Cmp"), ("Hea","Mar","Tck","Pos","Str"), ("Agg","Ant","Bra","Cnt","Dec")),
    "cds":  RoleDef("cds",  "Central Defender - Stopper",     ("Acc","Pac","Jum","Cmp"), ("Hea","Tck","Agg","Bra","Dec","Pos","Str"), ("Mar","Ant","Cnt")),
    "cdc":  RoleDef("cdc",  "Central Defender - Cover",       ("Acc","Pac","Jum","Cmp"), ("Mar","Tck","Ant","Cnt","Dec","Pos"), ("Hea","Bra","Str")),
    "ncbd": RoleDef("ncbd", "No-Nonsense CB - Defend",        ("Acc","Pac","Jum","Cmp"), ("Hea","Mar","Tck","Pos","Str"), ("Agg","Ant","Bra","Cnt")),
    "ncbs": RoleDef("ncbs", "No-Nonsense CB - Stopper",       ("Acc","Pac","Jum","Cmp"), ("Hea","Tck","Agg","Bra","Pos","Str"), ("Mar","Ant","Cnt")),
    "ncbc": RoleDef("ncbc", "No-Nonsense CB - Cover",         ("Acc","Pac","Jum","Cmp"), ("Mar","Tck","Ant","Cnt","Pos"), ("Hea","Bra","Str")),
    "ld":   RoleDef("ld",   "Libero - Defend",                ("Acc","Pac","Jum","Cmp"), ("Fir","Hea","Mar","Pas","Tck","Tec","Dec","Pos","Tea","Str"), ("Ant","Bra","Cnt","Sta")),
    "ls":   RoleDef("ls",   "Libero - Support",               ("Acc","Pac","Jum","Cmp"), ("Fir","Hea","Mar","Pas","Tck","Tec","Dec","Pos","Tea","Str"), ("Dri","Ant","Bra","Cnt","Vis","Sta")),
    "wcbd": RoleDef("wcbd", "Wide Centre Back - Defend",      ("Acc","Pac","Jum","Cmp"), ("Hea","Mar","Tck","Pos","Str"), ("Dri","Fir","Pas","Tec","Agg","Ant","Bra","Cnt","Dec","Wor","Agi")),
    "wcbs": RoleDef("wcbs", "Wide Centre Back - Support",     ("Acc","Pac","Jum","Cmp"), ("Dri","Hea","Mar","Tck","Pos","Str"), ("Cro","Fir","Pas","Tec","Agg","Ant","Bra","Cnt","Dec","OtB","Wor","Agi","Sta")),
    "wcba": RoleDef("wcba", "Wide Centre Back - Attack",      ("Acc","Pac","Jum","Cmp"), ("Cro","Dri","Hea","Mar","Tck","OtB","Sta","Str"), ("Fir","Pas","Tec","Agg","Ant","Bra","Cnt","Dec","Pos","Wor","Agi")),

    # === Full/Wing Backs ===
    "fbd":  RoleDef("fbd",  "Full Back - Defend",             ("Acc","Pac","Sta","Wor"), ("Mar","Tck","Ant","Cnt","Pos"), ("Cro","Pas","Dec","Tea")),
    "fbs":  RoleDef("fbs",  "Full Back - Support",            ("Acc","Pac","Sta","Wor"), ("Mar","Tck","Ant","Cnt","Pos","Tea"), ("Cro","Dri","Pas","Tec","Dec")),
    "fba":  RoleDef("fba",  "Full Back - Attack",             ("Acc","Pac","Sta","Wor"), ("Cro","Mar","Tck","Ant","Pos","Tea"), ("Dri","Fir","Pas","Tec","Cnt","Dec","OtB","Agi")),
    "nfbd": RoleDef("nfbd", "No-Nonsense FB - Defend",        ("Acc","Pac","Sta","Wor"), ("Mar","Tck","Ant","Pos","Str"), ("Hea","Agg","Bra","Cnt","Tea")),
    "ifbd": RoleDef("ifbd", "Inverted Full Back - Defend",    ("Acc","Pac","Sta","Wor"), ("Hea","Mar","Tck","Pos","Str"), ("Dri","Fir","Pas","Tec","Agg","Ant","Bra","Cmp","Cnt","Dec","Agi","Jum")),
    "wbd":  RoleDef("wbd",  "Wing Back - Defend",             ("Acc","Pac","Sta","Wor"), ("Mar","Tck","Ant","Pos","Tea"), ("Cro","Dri","Fir","Pas","Tec","Cnt","Dec","OtB","Agi","Bal")),
    "wbs":  RoleDef("wbs",  "Wing Back - Support",            ("Acc","Pac","Sta","Wor"), ("Cro","Dri","Mar","Tck","OtB","Tea"), ("Fir","Pas","Tec","Ant","Cnt","Dec","Pos","Agi","Bal")),
    "wba":  RoleDef("wba",  "Wing Back - Attack",             ("Acc","Pac","Sta","Wor"), ("Cro","Dri","Tck","Tec","OtB","Tea"), ("Fir","Mar","Pas","Ant","Cnt","Dec","Fla","Pos","Agi","Bal")),
    "cwbs": RoleDef("cwbs", "Complete Wing Back - Support",   ("Acc","Pac","Sta","Wor"), ("Cro","Dri","Tec","OtB","Tea"), ("Fir","Mar","Pas","Tck","Ant","Dec","Fla","Pos","Agi","Bal")),
    "cwba": RoleDef("cwba", "Complete Wing Back - Attack",    ("Acc","Pac","Sta","Wor"), ("Cro","Dri","Tec","Fla","OtB","Tea"), ("Fir","Mar","Pas","Tck","Ant","Dec","Pos","Agi","Bal")),
    "iwbd": RoleDef("iwbd", "Inverted Wing Back - Defend",    ("Acc","Pac","Sta","Wor"), ("Pas","Tck","Ant","Dec","Pos","Tea"), ("Fir","Mar","Tec","Cmp","Cnt","OtB","Agi")),
    "iwbs": RoleDef("iwbs", "Inverted Wing Back - Support",   ("Acc","Pac","Sta","Wor"), ("Fir","Pas","Tck","Cmp","Dec","Tea"), ("Mar","Tec","Ant","Cnt","OtB","Pos","Vis","Agi")),
    "iwba": RoleDef("iwba", "Inverted Wing Back - Attack",    ("Acc","Pac","Sta","Wor"), ("Fir","Pas","Tck","Tec","Cmp","Dec","OtB","Tea","Vis"), ("Cro","Dri","Lon","Mar","Ant","Cnt","Fla","Pos","Agi")),

    # === Defensive / Central Midfielders ===
    "ad":   RoleDef("ad",   "Anchor - Defend",                ("Wor","Sta","Acc","Pac"), ("Mar","Tck","Ant","Cnt","Dec","Pos"), ("Cmp","Tea","Str")),
    "hbd":  RoleDef("hbd",  "Half Back - Defend",             ("Wor","Sta","Acc","Pac"), ("Mar","Tck","Ant","Cmp","Cnt","Dec","Pos","Tea"), ("Fir","Pas","Agg","Bra","Jum","Str")),
    "dmd":  RoleDef("dmd",  "Defensive Midfielder - Defend",  ("Wor","Sta","Acc","Pac"), ("Tck","Ant","Cnt","Pos","Tea"), ("Mar","Pas","Agg","Cmp","Str","Dec")),
    "dms":  RoleDef("dms",  "Defensive Midfielder - Support", ("Wor","Sta","Acc","Pac"), ("Tck","Ant","Cnt","Pos","Tea"), ("Fir","Mar","Pas","Agg","Cmp","Dec","Str")),
    "bwmd": RoleDef("bwmd", "Ball Winning Mid - Defend",      ("Wor","Sta","Acc","Pac"), ("Tck","Agg","Ant","Tea"), ("Mar","Bra","Cnt","Pos","Agi","Str")),
    "bwms": RoleDef("bwms", "Ball Winning Mid - Support",     ("Wor","Sta","Acc","Pac"), ("Tck","Agg","Ant","Tea"), ("Mar","Pas","Bra","Cnt","Agi","Str")),
    "cmd":  RoleDef("cmd",  "Central Midfielder - Defend",    ("Acc","Pac","Sta","Wor"), ("Tck","Cnt","Dec","Pos","Tea"), ("Fir","Mar","Pas","Tec","Agg","Ant","Cmp")),
    "cms":  RoleDef("cms",  "Central Midfielder - Support",   ("Acc","Pac","Sta","Wor"), ("Fir","Pas","Tck","Dec","Tea"), ("Tec","Ant","Cmp","Cnt","OtB","Vis")),
    "cma":  RoleDef("cma",  "Central Midfielder - Attack",    ("Acc","Pac","Sta","Wor"), ("Fir","Pas","Dec","OtB"), ("Lon","Tck","Tec","Ant","Cmp","Tea","Vis")),
    "b2bs": RoleDef("b2bs", "Box-to-Box Midfielder - Support",("Acc","Pac","Sta","Wor"), ("Pas","Tck","OtB","Tea"), ("Dri","Fin","Fir","Lon","Tec","Agg","Ant","Cmp","Dec","Pos","Bal","Str")),
    "cars": RoleDef("cars", "Carrilero - Support",            ("Wor","Sta","Acc","Pac"), ("Fir","Pas","Tck","Dec","Pos","Tea"), ("Tec","Ant","Cmp","Cnt","OtB","Vis")),
    "dlpd": RoleDef("dlpd", "Deep Lying Playmaker - Defend",  ("Wor","Sta","Acc","Pac"), ("Fir","Pas","Tec","Cmp","Dec","Tea","Vis"), ("Tck","Ant","Pos","Bal")),
    "dlps": RoleDef("dlps", "Deep Lying Playmaker - Support", ("Wor","Sta","Acc","Pac"), ("Fir","Pas","Tec","Cmp","Dec","Tea","Vis"), ("Ant","OtB","Pos","Bal")),
    "regs": RoleDef("regs", "Regista - Support",              ("Wor","Sta","Acc","Pac"), ("Fir","Pas","Tec","Cmp","Dec","Fla","OtB","Tea","Vis"), ("Dri","Lon","Ant","Bal")),
    "rps":  RoleDef("rps",  "Roaming Playmaker - Support",    ("Acc","Pac","Sta","Wor"), ("Fir","Pas","Tec","Ant","Cmp","Dec","OtB","Tea","Vis"), ("Dri","Lon","Cnt","Pos","Agi","Bal")),
    "svs":  RoleDef("svs",  "Segundo Volante - Support",      ("Wor","Sta","Acc","Pac"), ("Mar","Pas","Tck","OtB","Pos"), ("Fin","Fir","Lon","Ant","Cmp","Cnt","Dec","Bal","Str")),
    "sva":  RoleDef("sva",  "Segundo Volante - Attack",       ("Wor","Sta","Acc","Pac"), ("Fin","Lon","Pas","Tck","Ant","OtB","Pos"), ("Fir","Mar","Cmp","Cnt","Dec","Bal")),

    # === Attacking Midfielders / Wingers ===
    "aps":  RoleDef("aps",  "Advanced Playmaker - Support",   ("Acc","Pac","Sta","Wor"), ("Fir","Pas","Tec","Cmp","Dec","OtB","Tea","Vis"), ("Dri","Ant","Fla","Agi")),
    "apa":  RoleDef("apa",  "Advanced Playmaker - Attack",    ("Acc","Pac","Sta","Wor"), ("Fir","Pas","Tec","Cmp","Dec","OtB","Tea","Vis"), ("Dri","Ant","Fla","Agi")),
    "ams":  RoleDef("ams",  "Attacking Midfielder - Support", ("Acc","Pac","Sta","Wor"), ("Fir","Lon","Pas","Tec","Ant","Dec","Fla","OtB"), ("Dri","Cmp","Vis","Agi")),
    "ama":  RoleDef("ama",  "Attacking Midfielder - Attack",  ("Acc","Pac","Sta","Wor"), ("Dri","Fir","Lon","Pas","Tec","Ant","Dec","Fla","OtB"), ("Fin","Cmp","Vis","Agi")),
    "engs": RoleDef("engs", "Enganche - Support",             ("Acc","Pac","Sta","Wor"), ("Fir","Pas","Tec","Cmp","Dec","Vis"), ("Dri","Ant","Fla","OtB","Tea","Agi")),
    "ssa":  RoleDef("ssa",  "Shadow Striker - Attack",        ("Acc","Pac","Sta","Wor"), ("Dri","Fin","Fir","Ant","Cmp","OtB"), ("Pas","Tec","Cnt","Dec","Agi","Bal")),
    "ifs":  RoleDef("ifs",  "Inside Forward - Support",       ("Acc","Pac","Sta","Wor"), ("Dri","Fin","Fir","Tec","OtB","Agi"), ("Lon","Pas","Ant","Cmp","Fla","Vis","Bal")),
    "ifa":  RoleDef("ifa",  "Inside Forward - Attack",        ("Acc","Pac","Sta","Wor"), ("Dri","Fin","Fir","Tec","Ant","OtB","Agi"), ("Lon","Pas","Cmp","Fla","Bal")),
    "iws":  RoleDef("iws",  "Inverted Winger - Support",      ("Acc","Pac","Sta","Wor"), ("Cro","Dri","Pas","Tec","Agi"), ("Fir","Lon","Cmp","Dec","OtB","Vis","Bal")),
    "iwa":  RoleDef("iwa",  "Inverted Winger - Attack",       ("Acc","Pac","Sta","Wor"), ("Cro","Dri","Pas","Tec","Agi"), ("Fir","Lon","Ant","Cmp","Dec","Fla","OtB","Vis","Bal")),
    "ws":   RoleDef("ws",   "Winger - Support",               ("Acc","Pac","Sta","Wor"), ("Cro","Dri","Tec","Agi"), ("Fir","Pas","OtB","Bal")),
    "wa":   RoleDef("wa",   "Winger - Attack",                ("Acc","Pac","Sta","Wor"), ("Cro","Dri","Tec","Agi"), ("Fir","Pas","Ant","Fla","OtB","Bal")),
    "wps":  RoleDef("wps",  "Wide Playmaker - Support",       ("Acc","Pac","Sta","Wor"), ("Fir","Pas","Tec","Cmp","Dec","Tea","Vis"), ("Dri","OtB","Agi")),
    "wpa":  RoleDef("wpa",  "Wide Playmaker - Attack",        ("Acc","Pac","Sta","Wor"), ("Dri","Fir","Pas","Tec","Cmp","Dec","OtB","Tea","Vis"), ("Ant","Fla","Agi")),
    "wmd":  RoleDef("wmd",  "Wide Midfielder - Defend",       ("Acc","Pac","Sta","Wor"), ("Pas","Tck","Cnt","Dec","Pos","Tea"), ("Cro","Fir","Mar","Tec","Ant","Cmp")),
    "wms":  RoleDef("wms",  "Wide Midfielder - Support",      ("Acc","Pac","Sta","Wor"), ("Pas","Tck","Dec","Tea"), ("Cro","Fir","Tec","Ant","Cmp","Cnt","OtB","Pos","Vis")),
    "wma":  RoleDef("wma",  "Wide Midfielder - Attack",       ("Acc","Pac","Sta","Wor"), ("Cro","Fir","Pas","Dec","Tea"), ("Tck","Tec","Ant","Cmp","OtB","Vis")),
    "dwd":  RoleDef("dwd",  "Defensive Winger - Defend",      ("Acc","Pac","Sta","Wor"), ("Tec","Ant","OtB","Pos","Tea"), ("Cro","Dri","Fir","Mar","Tck","Agg","Cnt","Dec")),
    "dws":  RoleDef("dws",  "Defensive Winger - Support",     ("Acc","Pac","Sta","Wor"), ("Cro","Pas","Tec","OtB","Tea"), ("Dri","Fir","Mar","Pas","Tck","Agg","Ant","Cmp","Cnt","Dec","Pos")),
    "mezs": RoleDef("mezs", "Mezzala - Support",              ("Acc","Pac","Sta","Wor"), ("Pas","Tec","Dec","OtB"), ("Dri","Fir","Lon","Tck","Ant","Cmp","Vis","Bal")),
    "meza": RoleDef("meza", "Mezzala - Attack",               ("Acc","Pac","Sta","Wor"), ("Dri","Pas","Tec","Dec","OtB","Vis"), ("Fin","Fir","Lon","Ant","Cmp","Fla","Bal")),
    "raua": RoleDef("raua", "Raumdeuter - Attack",            ("Acc","Pac","Sta","Wor"), ("Fin","Ant","Cmp","Cnt","Dec","OtB","Bal"), ("Fir","Tec")),

    # === Forwards ===
    "afa":  RoleDef("afa",  "Advanced Forward - Attack",      ("Acc","Pac","Fin"), ("Dri","Fir","Tec","Cmp","OtB"), ("Pas","Ant","Dec","Wor","Agi","Bal","Sta")),
    "cfs":  RoleDef("cfs",  "Complete Forward - Support",     ("Acc","Pac","Fin"), ("Dri","Fir","Hea","Lon","Pas","Tec","Ant","Cmp","Dec","OtB","Vis","Agi","Str"), ("Tea","Wor","Bal","Jum","Sta")),
    "cfa":  RoleDef("cfa",  "Complete Forward - Attack",      ("Acc","Pac","Fin"), ("Dri","Fir","Hea","Tec","Ant","Cmp","OtB","Agi","Str"), ("Lon","Pas","Dec","Tea","Vis","Wor","Bal","Jum","Sta")),
    "dlfs": RoleDef("dlfs", "Deep Lying Forward - Support",   ("Acc","Pac","Fin"), ("Fir","Pas","Tec","Cmp","Dec","OtB","Tea"), ("Ant","Fla","Vis","Bal","Str")),
    "dlfa": RoleDef("dlfa", "Deep Lying Forward - Attack",    ("Acc","Pac","Fin"), ("Fir","Pas","Tec","Cmp","Dec","OtB","Tea"), ("Dri","Ant","Fla","Vis","Bal","Str")),
    "f9s":  RoleDef("f9s",  "False Nine - Support",           ("Acc","Pac","Fin"), ("Dri","Fir","Pas","Tec","Cmp","Dec","OtB","Vis","Agi"), ("Ant","Fla","Tea","Bal")),
    "pa":   RoleDef("pa",   "Poacher - Attack",               ("Acc","Pac","Fin"), ("Ant","Cmp","OtB"), ("Fir","Hea","Tec","Dec")),
    "pfd":  RoleDef("pfd",  "Pressing Forward - Defend",      ("Acc","Pac","Fin"), ("Agg","Ant","Bra","Dec","Tea","Wor","Sta"), ("Fir","Cmp","Cnt","Agi","Bal","Str")),
    "pfs":  RoleDef("pfs",  "Pressing Forward - Support",     ("Acc","Pac","Fin"), ("Agg","Ant","Bra","Dec","Tea","Wor","Sta"), ("Fir","Pas","Cmp","Cnt","OtB","Agi","Bal","Str")),
    "pfa":  RoleDef("pfa",  "Pressing Forward - Attack",      ("Acc","Pac","Fin"), ("Agg","Ant","Bra","OtB","Tea","Wor","Sta"), ("Fir","Cmp","Cnt","Dec","Agi","Bal","Str")),
    "tfs":  RoleDef("tfs",  "Target Forward - Support",       ("Acc","Pac","Fin"), ("Hea","Bra","Tea","Bal","Jum","Str"), ("Fir","Agg","Ant","Cmp","Dec","OtB")),
    "tfa":  RoleDef("tfa",  "Target Forward - Attack",        ("Acc","Pac","Fin"), ("Hea","Bra","Cmp","OtB","Bal","Jum","Str"), ("Fir","Agg","Ant","Dec","Tea")),
    "trea": RoleDef("trea", "Trequartista - Attack",          ("Acc","Pac","Fin"), ("Dri","Fir","Pas","Tec","Cmp","Dec","Fla","OtB","Vis"), ("Ant","Agi","Bal")),
    "wtfs": RoleDef("wtfs", "Wide Target Forward - Support",  ("Acc","Pac","Sta","Wor"), ("Hea","Bra","Tea","Jum","Str"), ("Cro","Fir","Ant","OtB","Bal")),
    "wtfa": RoleDef("wtfa", "Wide Target Forward - Attack",   ("Acc","Pac","Sta","Wor"), ("Hea","Bra","OtB","Jum","Str"), ("Cro","Fin","Fir","Ant","Tea","Bal")),
}
# fmt: on

# Group roles by position for UI display
ROLE_GROUPS = {
    "Goalkeepers": ["gkd", "skd", "sks", "ska"],
    "Defenders": ["bpdd", "bpds", "bpdc", "cdd", "cds", "cdc", "ncbd", "ncbs", "ncbc", "ld", "ls", "wcbd", "wcbs", "wcba"],
    "Full/Wing Backs": ["fbd", "fbs", "fba", "nfbd", "ifbd", "wbd", "wbs", "wba", "cwbs", "cwba", "iwbd", "iwbs", "iwba"],
    "Defensive/Central Mid": ["ad", "hbd", "dmd", "dms", "bwmd", "bwms", "cmd", "cms", "cma", "b2bs", "cars", "dlpd", "dlps", "regs", "rps", "svs", "sva"],
    "Attacking Mid/Wingers": ["aps", "apa", "ams", "ama", "engs", "ssa", "ifs", "ifa", "iws", "iwa", "ws", "wa", "wps", "wpa", "wmd", "wms", "wma", "dwd", "dws", "raua"],
    "Forwards": ["afa", "cfs", "cfa", "dlfs", "dlfa", "f9s", "pa", "pfd", "pfs", "pfa", "tfs", "tfa", "trea", "wtfs", "wtfa"],
}
