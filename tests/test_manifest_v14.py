"""Turn 14 Phase 0: an archived attempt cannot overwrite a scored episode."""

import json

import pytest

from statebench import manifest_v14 as M


def _rec(seed, success, cond="T", reader="x", error=None):
    return {"reader": reader, "condition": cond, "seed": seed, "success": success,
            "steps": [{"error": error}]}


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _manifest(seeds=(1, 2)):
    return {"t": {"confirm": {"x": {"file": "t/confirm_llm_x.jsonl", "conditions": ("T",), "seeds": seeds,
                                    "archived": ["t/confirm_llm_x_transport_errors.jsonl"]}}}}


def test_archived_attempt_cannot_overwrite_scored(tmp_path):
    _write(tmp_path / "t/confirm_llm_x.jsonl", [_rec(1, True), _rec(2, True)])
    _write(tmp_path / "t/confirm_llm_x_transport_errors.jsonl", [_rec(1, False, error="http 429: rate limit")])
    got = M.load("t", "confirm", "x", _manifest(), tmp_path)
    assert got["T"][1]["success"] is True and got["T"][2]["success"] is True
    # the Turn 13 loader's behaviour (glob every confirm_llm_*.jsonl, later files overwrite): the bug this prevents
    by = {}
    for f in sorted((tmp_path / "t").glob("confirm_llm_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            by[(r["condition"], r["seed"])] = r
    assert by[("T", 1)]["success"] is False
    assert len(M.archived("t", "confirm", "x", _manifest(), tmp_path)) == 1


def test_duplicate_record_rejected(tmp_path):
    _write(tmp_path / "t/confirm_llm_x.jsonl", [_rec(1, True), _rec(1, False), _rec(2, True)])
    with pytest.raises(M.ManifestError, match="second record"):
        M.load("t", "confirm", "x", _manifest(), tmp_path)


def test_missing_seed_rejected(tmp_path):
    _write(tmp_path / "t/confirm_llm_x.jsonl", [_rec(1, True)])
    with pytest.raises(M.ManifestError, match="seed set differs"):
        M.load("t", "confirm", "x", _manifest(), tmp_path)


def test_transport_error_in_scored_file_rejected(tmp_path):
    _write(tmp_path / "t/confirm_llm_x.jsonl", [_rec(1, True), _rec(2, False, error="http 529: overloaded")])
    with pytest.raises(M.ManifestError, match="transport error"):
        M.load("t", "confirm", "x", _manifest(), tmp_path)
