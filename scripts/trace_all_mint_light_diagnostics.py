"""Run resumable 10-episode MINT-Light diagnostics over LIBERO-130."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


SUITES = {"libero_object": 10, "libero_spatial": 10, "libero_goal": 10,
          "libero_10": 10, "libero_90": 90}


def summarize(output_dir):
    tasks = []
    for suite, count in SUITES.items():
        for task_id in range(count):
            name = f"{suite}_{task_id:02}"
            task_dir = output_dir / name
            trace_path = task_dir / "diagnostic_trace.json"
            videos = list((task_dir / "annotated_videos").glob("episode_*.mp4"))
            if not trace_path.exists() or len(videos) != 10:
                continue
            trace = json.loads(trace_path.read_text())
            if len(trace["episodes"]) != 10:
                continue
            tasks.append({"suite": suite, "task_id": task_id,
                          "successes": sum(episode["success"] for episode in trace["episodes"]),
                          "actions": [episode["n_actions"] for episode in trace["episodes"]],
                          "videos": str(task_dir / "annotated_videos")})
    summary = {"completed_tasks": len(tasks), "total_tasks": 130,
               "completed_episodes": len(tasks) * 10,
               "successes": sum(task["successes"] for task in tasks), "tasks": tasks}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {f"{task['suite']}_{task['task_id']:02}" for task in tasks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-length", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tracer = Path(__file__).with_name("trace_mint_light_diagnostics.py")
    renderer = Path(__file__).with_name("render_diagnostic_videos.py")
    for suite, count in SUITES.items():
        for task_id in range(count):
            name = f"{suite}_{task_id:02}"
            task_dir = output_dir / name
            if name in summarize(output_dir):
                continue
            task_dir.mkdir(exist_ok=True)
            trace_path = task_dir / "diagnostic_trace.json"
            if trace_path.exists() and len(json.loads(trace_path.read_text())["episodes"]) == 10:
                command = [sys.executable, str(renderer), str(task_dir)]
            else:
                command = [sys.executable, str(tracer), "--checkpoint", str(args.checkpoint),
                           "--suite", suite, "--task-id", str(task_id), "--seed", str(args.seed),
                           "--n-episodes", "10", "--episode-length", str(args.episode_length),
                           "--annotate-videos", "--output-dir", str(task_dir)]
            print(f"Running {name}: {'annotation' if command[1] == str(renderer) else 'rollout'}", flush=True)
            with (task_dir / "run.log").open("a") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            summarize(output_dir)
    print("LIBERO-130 evaluation complete", flush=True)


if __name__ == "__main__":
    main()
