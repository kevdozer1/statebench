# statebench

statebench asks what a robot's state must contain, as text, for a language model to act on it. A simulated scene is
written as tracked positions, a menu of skills, a few yes/no labels and a goal. A fast planner (Jev) reads
it and picks the next skill. A slower model (Sol) writes what no script can, like the goal or every stroke of a
drawing. Over 19 rounds I removed and added pieces of the state and measured what changed.

Write-up: [How chatbotics works](https://kevdozer1.com/blog/2026/training-chatbots-to-use-robots/). Paper: [PDF](https://kevdozer1.com/assets/pdf/statebench_paper.pdf), with its source in [paper/](paper/).

## What it found

- **Positions and a complete goal were enough.** Jev finished pick and place 50 of 50 times. Adding meaning labels and
  procedure fields dropped it to 45, because it looped.
- **The goal's ending mattered most.** With the ending left out, Jev finished 17 of 50 on positions. The extra fields
  patched that up to 47. With no goal at all it finished 49.
- **Fine contact needs a real sensor.** A false "pen not touching" knocked the pen out of the hand in 50 of 50
  drawings. A false "touching" did no harm.
- **Goals written from video copy the video.** Sol stated the final withdrawal in 1 of 50 goals and the pen lift in 15.
  A finish skill that lifts the pen fixed the drawings without touching the goal.
- **Splitting a drawing into rounds hurt.** In blind ratings from 1 to 5, one shot scored 2.25 and four rounds 1.08.
  Rounds that each drew one part inside a box planned up front came back to 2.17.

More in [results/RESULTS.md](results/RESULTS.md) and [docs/HISTORY.md](docs/HISTORY.md).

## What it does not show

Everything is simulated in MuJoCo, on tasks built for this project. The models were used frozen through their APIs,
and nothing was trained on these labels. The drawing ratings come from one rater (me), who had seen previews. Jev and
Sol were tested separately, and the full loop has not been run.

## Install

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/kevdozer1/statebench && cd statebench
uv sync --extra dev
uv run pytest
```

Pick and place uses the xArm7 from MuJoCo Menagerie. The drawing robot runs in
[llm-robotics-playground](https://github.com/dimentary/llm-robotics-playground), in its own environment:

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie data/mujoco_menagerie
git clone https://github.com/dimentary/llm-robotics-playground data/llm-robotics-playground
cd data/llm-robotics-playground && uv sync --locked && cd ../..
```

Keys go in the environment or a `.env` file: `OPENROUTER_API_KEY` for Sol, `TYPESAFE_JEV_API_KEY` for Jev. Other
paths are in [config.py](src/statebench/config.py).

## Quickstart

```bash
uv run statebench draw --painting mona_lisa --way one-shot      # one Sol drawing, about 3 cents
uv run statebench pick-place --seed 1600 --goal no-ending        # one Jev pick and place episode
uv run statebench fields --seeds 1600:1610                       # success by goal and state fields
```

Without a key, `draw` prints the prompt Sol would get, and the Jev commands run a scripted reader instead. Both say so.

## Where things are

| In `src/statebench/` | What |
|---|---|
| `draw18.py`, `draw18a5.py`, `draw19.py` | Sol draws, five ways plus two round variants |
| `runner_v14.py`, `prompt_v14.py`, `readers_v14.py` | pick and place with Jev, by state fields |
| `tasks15.py`, `tasks17.py`, `grid_v17.py`, `audit16.py` | tracing, button and block, goals from video |
| `stats_v14.py`, `hashing.py`, `manifest_v14.py` | paired statistics, hashes, a strict results loader |

The scene servers in `scenes/` run inside llm-robotics-playground. Module names keep their round number. Later rounds subclassed earlier modules instead of editing them, because each
round's results were pinned to file hashes.

## License

Apache-2.0. Third-party credits are in [THIRD_PARTY.md](THIRD_PARTY.md).
