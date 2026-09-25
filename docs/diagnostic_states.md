# Diagnostic state tracing

`libero.libero.envs.diagnostic_states.DiagnosticStateTracker` adds diagnostic
labels without changing LIBERO's BDDL goal predicates or success checks.
Call `reset()` after the simulator's settling steps for each episode, then
`sample(task_atoms)` after each action.

The tracker records an episode-specific position and orientation baseline for
each movable object. `initial_orientation` compares the object's local vertical
axis with that baseline, ignoring yaw. `fallen` additionally requires a lower
center height, contact with another scene object, and no detected grasp. These
are diagnostic heuristics; a moved or tilted object can satisfy neither label.

`holding` requires both gripper fingerpad groups to contact the object for two
successive samples, using robosuite's grasp check. `on_center` and `in_center`
are evaluated only when LIBERO's corresponding `on` or `in` predicate is true.
For sites they use half of the target's local XY extent by default; for object
supports, `on_center` uses half of LIBERO's 0.03 m XY threshold. Drawer labels
`closed`, `partially_open`, and `fully_open` use the object's joint range;
`fully_open` starts at 90% of the declared travel by default, while the first
5% is labeled `closed` to handle strict BDDL threshold boundaries. All
thresholds are configurable through `DiagnosticThresholds`.

To run a MINT-Light diagnostic rollout from this checkout with the MINT Python
environment and checkpoint:

```bash
PYTHONPATH=$PWD /path/to/MINT/.venv/bin/python scripts/trace_mint_light_diagnostics.py \
  --checkpoint /path/to/checkpoint --suite libero_90 --task-id 13 \
  --seed 1000 --n-episodes 8 --episode-length 400
```

The script writes `diagnostic_trace.json` beside LeRobot's `eval_info.json` and
videos. `--episode-length` overrides LeRobot's suite limit. For a rollout that
originally occupied index `i` in a batch, rerun episodes `0..i` from the same
starting seed to preserve the initial-state sequence. Even then, GPU policy
sampling may differ from a previous run, so compare recorded outcomes rather
than assuming exact trajectory replay.
