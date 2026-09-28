"""The model-draws task: prompts match what Sol was sent in the rated runs, the program parser, the canvas mapping."""
from __future__ import annotations

import hashlib
import json

from statebench import draw18 as A
from statebench import draw18a5 as A5
from statebench import draw19 as Q
from statebench import draws16 as D
from statebench import parts17 as P


def sha(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").encode()).hexdigest()


def test_templates_match_the_frozen_hashes():
    h, h5, h19 = A.prompt_hashes(), A5.prompt_hashes(), Q.prompt_hashes()
    assert h["A1"] == "ee0f58bd8ef1cdc45d7cf8641874b306b5f4f8ff77b0e159172dadc4ce83c850"
    assert h["A2"] == "68d5363d13d159399ed661cf8d65e3a11a16f2e254c1af4959812e58727036c1"
    assert h["A3"] == "61f7325483d5a2241fbf35799f26a33c75cea832d240e73eb8dbf37535bc6e8e"
    assert h["A4"] == "5c268ceaad4a00b1bb1ad6a491b5b2cf8db4b7410cbe1d5b65093f8aef21c6a1"
    assert h5["A5"] == "a1b6b38ad904145af3ba5b3de953917cdf5e59a97bf8b17656f023d931c7257d"
    assert h5["layout_schema"] == "e6004da7471136f6b8f13df03a66bf19957468e44501f76dbad240488800ecad"
    assert h19["A2c"] == "0a6032468f1cd6774b6ec52abd8e46292e68e8564c3de8e56c5a790d5a72a647"
    assert h19["R"] == "96b84ea9d0442ea02ac28ca1704bdfe634a7c9f130975e800357b8d8e70540c2"


def test_full_prompts_match_the_rated_runs():
    # prompt_sha256 of round 1, sample 0, in the rated Mona Lisa and Starry Night drawings
    assert sha(A.prompt_a1("mona_lisa")) == "1444b384319e29b28a0c9bf6004149a6c221efd27ef7f2bf7a9392c119fbdf17"
    assert sha(A.prompt_a1("starry_night")) == "1280fa2e19ab6e190d1a4c8cf5c577ade21a253514b328092f61a7405bf7d420"
    assert sha(A.prompt_round("mona_lisa", "A3", 1)) == "a6e28b01f4edd92dfd2a2480a2460aa9958e54c534035e576001bcb2d2711ab9"
    assert sha(A.prompt_round("starry_night", "A3", 1)) == "5bf86462c5dad5fbf9bb7cecc11e5c9216bf0b3a69ceddeb6659413356615c70"


def test_parser_limits_and_clipping():
    reply = json.dumps({"actions": [
        {"type": "draw", "points": [[10, 10], [20, 20], [95, 30]]},
        {"type": "draw", "points": [[5, 5], [5, 5]]},
        {"type": "lift", "points": []},
        {"type": "finish", "points": []},
        {"type": "draw", "points": [[1, 1], [2, 2]]},
    ]})
    ops, counts = A.parse(reply, 60)
    assert [o[0] for o in ops] == ["draw", "lift", "finish"]
    assert ops[0][1][-1] == [90.0, 30.0]
    assert counts["points_off_canvas"] == 1
    assert counts["degenerate_strokes"] == 1
    assert counts["after_finish_ignored"] == 1
    ops, counts = A.parse("not json", 60)
    assert ops == [] and counts["unparseable"] == 1


def test_parser_keeps_at_most_the_allowed_strokes():
    reply = json.dumps({"actions": [{"type": "draw", "points": [[i, 1], [i, 2]]} for i in range(1, 21)]})
    ops, counts = A.parse(reply, 15)
    assert sum(o[0] == "draw" for o in ops) == 15
    assert counts["strokes_dropped_over_max"] == 5


def test_canvas_fits_the_painting_inside_a_5_mm_margin():
    for ref in P.PAINTINGS:
        x0, y0, w, h = P.fit(ref)
        assert min(x0, y0) >= 5 - 1e-9
        assert max(w, h) == 80.0
        assert abs(x0 + w + x0 - D.CANVAS_MM) < 1e-9 and abs(y0 + h + y0 - D.CANVAS_MM) < 1e-9
