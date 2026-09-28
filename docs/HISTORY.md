# How the question changed

statebench ran as 19 rounds. Each round's prompts and analysis rules were hashed before the runs that tested them, and
nothing pinned was edited afterwards. That is why module names carry a round number.

**Rounds 1 to 4: does the wording of the state matter?** A simulated xArm7 puts a cube in a tray, and the same scene was
written out several ways. Jev scored 0 of 50 on a plain caption and 38 of 50 on structured fields. Adding computed
yes/no labels, then correcting one sentence in them, took it to 50 of 50.

**Rounds 5 to 8: noisy state.** Positions estimated from a camera lost nothing. A 1 degree camera tilt or 150 ms of
latency broke the rule-based planner until fixes that use no ground truth brought it back. Round 8 set the noise
levels used later (the harder one is a 5 mm bias plus 2 mm jitter).

**Rounds 9 to 12: which labels matter, and a pen.** A drawing scene was added, built on llm-robotics-playground. A false
"not touching" knocked the pen out of the hand every time, and a false "touching" did no harm. Round 12 fixed a clock
that ran at 25 Hz instead of 30 and caught a tilted page that never tilted.

**Rounds 13 and 14: language models read the state.** With a goal that left out the ending, Jev finished 17 of 50 on
positions, 28 with meaning labels and 47 with procedure fields too. Round 14 wrote the ending into the goal, and
positions alone went to 50 of 50.

**Rounds 15 to 17: where does the goal come from?** Sol wrote goals from video, and most left out the ending. With no
goal at all Jev still finished the block task 49 of 50, because the skill menu implied it. A finish skill that lifts
the pen closed the drawing gap.

**Rounds 18 and 19: the model draws.** Sol looks at a painting and writes every stroke. I compared five ways of giving
it the job and rated the drawings blind. One shot won, and rounds hurt unless each round got its own place on the page.
Giving Sol the coordinates of its earlier strokes helped rounds on the automatic scores. Those drawings are not rated
yet.

## Mistakes that were caught

**The loader bug (round 13, found in round 14).** The results loader matched files by pattern, and the pattern also
caught archived files of failed network attempts. They sorted last and overwrote the successful re-runs. An outside
review found it. The fix moved Sonnet from 45 to 50 of 50 and withdrew one comparison. Every later result goes through
`manifest_v14.py`, which refuses duplicates, with a regression test.

**The "hold" that was a loop (round 17).** A contrasting goal asked Jev to lift the cube and hold it above the tray, and
49 of 50 episodes passed the hold check, which reads the final state. None of them held anything. All 49 ran the same
24 steps: grasp, lift, carry and release four times over, then a fifth grasp and carry that the decision cap cut off
mid-air. Holding was reported as not demonstrated.

**The tracing planner already held the task (rounds 15 to 17).** In the tracing task the planner saw a stroke list made
by a script, so the list was the task. Every failed tracing failed the same way, with the pen left down at the end, and
goal wording could not show much. This is why round 18 moved to drawings where Sol writes every stroke itself.
