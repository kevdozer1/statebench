"""Turn 14 Phase 0: selection of scored LLM episodes by explicit manifest.

Turn 13's report (``confirm_v13._llm_rows``) globbed ``confirm_llm_*.jsonl``, which also matched the archived
``confirm_llm_<reader>_transport_errors.jsonl`` files. They sort after the scored files, so their old records
overwrote the successful re-runs (Sonnet T read 45/50 instead of 50/50; Jev T 16/50 instead of 17/50).

From now on every scored LLM file is named here, per turn, phase and reader, with its conditions and seed set. The
loader reads only the named file and asserts:
* every record belongs to the named reader (a file the manifest marks ``shared`` may hold other readers' records,
  which are skipped) and to one of the named conditions;
* exactly one record per (condition, seed), and exactly the expected seed set per condition;
* no record contains a transport error step (an error starting with "http").
Archived attempts are listed separately (``archived``) and are read only by the transport-reliability report.
"""

from __future__ import annotations

import json
from pathlib import Path
from .config import RUNS

ROOT = RUNS
T13_CONFIRM_SEEDS = tuple(range(1500, 1600, 2))
T13_DEV_SEEDS = tuple(range(0, 200, 10))
C5 = ("T", "T+M", "T+M+P", "T+M late", "T+M flip")

MANIFEST: dict = {
    "turn13": {
        "confirm": {
            "local-small": {"file": "turn13/confirm_llm_local-small.jsonl", "conditions": C5, "seeds": T13_CONFIRM_SEEDS},
            "local-mid": {"file": "turn13/confirm_llm_local-mid.jsonl", "conditions": C5, "seeds": T13_CONFIRM_SEEDS},
            "jev": {"file": "turn13/confirm_llm_jev.jsonl", "conditions": C5, "seeds": T13_CONFIRM_SEEDS,
                    "archived": ["turn13/confirm_llm_jev_transport_errors.jsonl"]},
            "sonnet": {"file": "turn13/confirm_llm_sonnet.jsonl", "conditions": ("T", "T+M", "T+M+P", "T+M late"),
                       "seeds": T13_CONFIRM_SEEDS, "archived": ["turn13/confirm_llm_sonnet_transport_errors.jsonl"]},
        },
        "dev": {
            "rules_v2": {"file": "turn13/p2_dev.jsonl", "conditions": ("T+M+P",), "seeds": T13_DEV_SEEDS, "shared": True},
            "local-small": {"file": "turn13/p2_dev.jsonl", "conditions": C5, "seeds": T13_DEV_SEEDS, "shared": True},
            "local-mid": {"file": "turn13/p2_dev_mid.jsonl", "conditions": C5 + ("T+M truth",), "seeds": T13_DEV_SEEDS},
            "jev": {"file": "turn13/p2_dev_jev.jsonl", "conditions": C5, "seeds": T13_DEV_SEEDS},
            "sonnet": {"file": "turn13/sonnet_probe.jsonl", "conditions": ("T+M+P",), "seeds": (0, 10, 20)},
            "astra": {"file": "turn13/astra_probe.jsonl", "conditions": ("T+M+P",), "seeds": (0, 10),
                      "archived": ["turn13/astra_probe_rejected_reasoning_off.jsonl"]},
        },
    },
}


T14_DEV_PP = tuple(range(0, 200, 10))
T14_DEV_BUTTON = tuple(range(20))
T14_CONFIRM = tuple(range(1600, 1700, 2))
PP_ALL = ("T", "T+H", "T+M", "T+M+P", "T+M+P-A", "T+M+P-S", "T+M+P-C", "T+M+A", "T+M+S", "T+M+C")
MANIFEST["turn14"] = {
    "diag": {
        "local-small": {"file": "turn14/p1_diag.jsonl", "conditions": ("T+M",), "seeds": T14_DEV_PP, "shared": True},
        "local-mid": {"file": "turn14/p1_diag.jsonl", "conditions": ("T+M",), "seeds": T14_DEV_PP, "shared": True},
    },
    "dev_pp": {
        "rules": {"file": "turn14/p2_dev_rules.jsonl", "conditions": ("T+H", "T+M+P"), "seeds": T14_DEV_PP},
        "jev": {"file": "turn14/p2_dev_jev.jsonl", "conditions": PP_ALL, "seeds": T14_DEV_PP},
        "local-mid": {"file": "turn14/p2_dev_mid.jsonl", "conditions": ("T", "T+H", "T+M", "T+M+P"), "seeds": T14_DEV_PP},
        "sonnet": {"file": "turn14/p2_dev_sonnet.jsonl", "conditions": ("T", "T+H"), "seeds": T14_DEV_PP},
    },
    "dev_button": {
        "rules": {"file": "turn14/p3_dev_rules.jsonl", "conditions": ("T+H", "T+M+P"), "seeds": T14_DEV_BUTTON},
        "jev": {"file": "turn14/p3_dev_jev.jsonl", "conditions": ("T", "T+H", "T+M", "T+M+P", "T+M+P-A", "T+M+P-S", "T+M+P-C"),
                "seeds": T14_DEV_BUTTON},
        "local-mid": {"file": "turn14/p3_dev_mid.jsonl", "conditions": ("T", "T+H", "T+M", "T+M+P"), "seeds": T14_DEV_BUTTON},
        "sonnet": {"file": "turn14/p3_dev_sonnet.jsonl", "conditions": ("T", "T+M+P"), "seeds": T14_DEV_BUTTON},
    },
}

CONFIRM_CELLS = {
    "pick_and_place": {"jev": ("T", "T+H", "T+M+P", "T+M+P-A"), "rules": ("T+H",), "sonnet": ("T",),
                       "local-mid": ("T", "T+M")},
    "button": {"jev": ("T", "T+H", "T+M+P", "T+M+P-A"), "rules": ("T+H",), "sonnet": ("T+M+P",),
               "local-mid": ("T", "T+M")},
}
for _task, _cells in CONFIRM_CELLS.items():
    MANIFEST["turn14"][f"confirm_{_task}"] = {
        rd: {"file": f"turn14/confirm_{_task}_{rd}.jsonl", "conditions": conds, "seeds": T14_CONFIRM,
             "archived": [f"turn14/confirm_{_task}_{rd}_transport_errors.jsonl"]} for rd, conds in _cells.items()}


class ManifestError(AssertionError):
    pass


def has_transport_error(r: dict) -> bool:
    return any((s.get("error") or "").startswith("http") for s in r.get("steps", []))


def load(turn: str, phase: str, reader: str, manifest: dict | None = None, root: Path | None = None) -> dict:
    """{condition: {seed: record}} from the one scored file the manifest names."""
    m = (manifest or MANIFEST)[turn][phase][reader]
    root = Path(root) if root is not None else ROOT
    path = root / m["file"]
    out: dict = {c: {} for c in m["conditions"]}
    for i, line in enumerate(path.read_text().splitlines()):
        r = json.loads(line)
        if r.get("reader") != reader:
            if m.get("shared"):
                continue  # a file shared by readers (declared in the manifest); other readers' records are skipped
            raise ManifestError(f"{path.name}:{i + 1}: reader {r.get('reader')!r}, expected {reader!r}")
        c = r["condition"]
        if c not in out:
            raise ManifestError(f"{path.name}:{i + 1}: condition {c!r} is not in the manifest for {reader}")
        if r["seed"] in out[c]:
            raise ManifestError(f"{path.name}:{i + 1}: a second record for ({c}, {r['seed']})")
        if has_transport_error(r):
            raise ManifestError(f"{path.name}:{i + 1}: ({c}, {r['seed']}) contains a transport error")
        out[c][r["seed"]] = r
    for c, recs in out.items():
        if set(recs) != set(m["seeds"]):
            missing, extra = sorted(set(m["seeds"]) - set(recs)), sorted(set(recs) - set(m["seeds"]))
            raise ManifestError(f"{reader} {c}: seed set differs (missing {missing[:5]}, extra {extra[:5]})")
    return out


def archived(turn: str, phase: str, reader: str, manifest: dict | None = None, root: Path | None = None) -> list[dict]:
    m = (manifest or MANIFEST)[turn][phase][reader]
    root = Path(root) if root is not None else ROOT
    rows = []
    for f in m.get("archived", []):
        if (root / f).exists():
            rows += [json.loads(x) for x in (root / f).read_text().splitlines()]
    return rows
