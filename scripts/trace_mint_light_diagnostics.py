"""Run MINT-Light in LIBERO and save BDDL plus diagnostic state changes.

Run with the MINT environment's Python and this repository on PYTHONPATH.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def format_atom(atom):
    return "(" + " ".join(atom) + ")"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", default="libero_90")
    parser.add_argument("--task-id", type=int, default=13)
    parser.add_argument("--seed", type=int, default=1007)
    parser.add_argument("--init-state-index", type=int,
                        help="LIBERO initialization index used by the original batch episode")
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--episode-length", type=int)
    parser.add_argument("--annotate-videos", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/eval/mint_light_diagnostic_trace"))
    args = parser.parse_args()
    if args.n_episodes < 1 or (args.episode_length is not None and args.episode_length < 1):
        parser.error("episode counts and lengths must be positive")

    os.environ.update({"MUJOCO_GL": "egl", "HF_HUB_OFFLINE": "1",
                       "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})

    from libero.libero.envs.diagnostic_states import DiagnosticStateTracker
    from lerobot.envs.libero import LiberoEnv
    import lerobot.scripts.lerobot_eval as eval_module
    from lerobot.scripts.lerobot_eval import main as eval_main

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = {"suite": args.suite, "task_id": args.task_id, "seed": args.seed,
             "episode_length": args.episode_length, "episodes": []}
    original_reset = LiberoEnv.reset
    original_step = LiberoEnv.step
    original_init = LiberoEnv.__init__
    original_render = LiberoEnv.render
    original_write_video = eval_module.write_video
    final_frames = {}

    def traced_init(self, *init_args, **init_kwargs):
        if args.init_state_index is not None:
            init_kwargs["episode_index"] = args.init_state_index
        original_init(self, *init_args, **init_kwargs)

    def sample(env, step):
        raw = env._env.env
        episode = trace["episodes"][-1]
        true_atoms = {format_atom(atom) for atom in env._diagnostic_task_atoms
                      if raw._eval_predicate(atom)}
        true_atoms.update(format_atom(atom) for atom in env._diagnostic_tracker.sample(
            env._diagnostic_task_atoms))
        previous = env._diagnostic_previous
        if step == 0 or true_atoms != previous:
            episode["state_path"].append({"step": step, "added": sorted(true_atoms - previous),
                                          "removed": sorted(previous - true_atoms)})
        episode["n_actions"] = step
        episode["goal_satisfied_final"] = all(format_atom(atom) in true_atoms
                                               for atom in env._diagnostic_goal_atoms)
        env._diagnostic_previous = true_atoms

    def traced_reset(self, *reset_args, **reset_kwargs):
        result = original_reset(self, *reset_args, **reset_kwargs)
        if not getattr(self, "_diagnostic_in_step", False):
            raw = self._env.env
            problem = raw.parsed_problem
            if "instruction" not in trace:
                trace["instruction"] = self.task_description
                trace["bddl_file"] = Path(raw.bddl_file_name).name
                trace["goal_atoms"] = [format_atom(atom) for atom in problem["goal_state"]]
            self._diagnostic_goal_atoms = [tuple(atom) for atom in problem["goal_state"]]
            self._diagnostic_task_atoms = list(dict.fromkeys(
                tuple(atom) for atom in problem["initial_state"] + problem["goal_state"]
            ))
            self._diagnostic_tracker = DiagnosticStateTracker(raw)
            self._diagnostic_tracker.reset()
            self._diagnostic_previous = set()
            episode_index = len(trace["episodes"])
            trace["episodes"].append({"episode_index": episode_index,
                                      "seed": args.seed + episode_index, "state_path": []})
            sample(self, 0)
            self._diagnostic_step = 0
        return result

    def traced_step(self, action):
        raw_step = self._env.step
        captured = False

        def capture(raw_action):
            nonlocal captured
            result = raw_step(raw_action)
            if not captured:
                self._diagnostic_step += 1
                sample(self, self._diagnostic_step)
                if result[2]:
                    # LiberoEnv.step resets internally on success. Keep the terminal image.
                    self._diagnostic_terminal_frame = original_render(self)
                captured = True
            return result

        self._env.step = capture
        self._diagnostic_in_step = True
        try:
            return original_step(self, action)
        finally:
            self._env.step = raw_step
            self._diagnostic_in_step = False

    def traced_render(self):
        frame = getattr(self, "_diagnostic_terminal_frame", None)
        if frame is not None:
            self._diagnostic_terminal_frame = None
        else:
            frame = original_render(self)
        final_frames[len(trace["episodes"]) - 1] = frame
        return frame

    def write_complete_video(path, frames, fps, *args, **kwargs):
        episode_index = int(Path(path).stem.rsplit("_", 1)[1])
        final_frame = final_frames.get(episode_index)
        if final_frame is None:
            raise RuntimeError(f"Missing final frame for episode {episode_index}")
        complete_frames = np.concatenate((frames, final_frame[None]), axis=0)
        return original_write_video(path, complete_frames, fps, *args, **kwargs)

    LiberoEnv.__init__ = traced_init
    LiberoEnv.reset = traced_reset
    LiberoEnv.step = traced_step
    LiberoEnv.render = traced_render
    eval_module.write_video = write_complete_video
    sys.argv = [sys.argv[0], f"--policy.path={args.checkpoint.resolve()}",
                "--env.type=libero", f"--env.task={args.suite}",
                f"--env.task_ids=[{args.task_id}]", "--eval.batch_size=1",
                f"--eval.n_episodes={args.n_episodes}", f"--seed={args.seed}",
                f"--output_dir={output_dir}"]
    if args.episode_length is not None:
        sys.argv.append(f"--env.episode_length={args.episode_length}")
    try:
        eval_main()
    finally:
        result_file = output_dir / "eval_info.json"
        if result_file.exists():
            results = json.loads(result_file.read_text())["per_task"][0]["metrics"]["successes"]
            if len(results) != len(trace["episodes"]):
                raise RuntimeError("Episode count differs between evaluation and diagnostic trace")
            for episode, success in zip(trace["episodes"], results):
                episode["success"] = bool(success)
        trace_path = output_dir / "diagnostic_trace.json"
        trace_path.write_text(json.dumps(trace, indent=2) + "\n")
    if args.annotate_videos:
        from render_diagnostic_videos import annotate_task_videos

        annotate_task_videos(output_dir)


if __name__ == "__main__":
    main()
