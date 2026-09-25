"""Save raw video frames and full symbolic states at each state transition."""

import argparse
import json
from pathlib import Path

import cv2


DIAGNOSTIC_PREDICATES = {
    "initial_orientation", "fallen", "holding", "on_center", "in_center",
    "closed", "partially_open", "fully_open",
}


def _is_diagnostic(atom):
    return atom[1:].split(" ", 1)[0] in DIAGNOSTIC_PREDICATES


def export_state_change_frames(output_dir):
    output_dir = Path(output_dir)
    trace = json.loads((output_dir / "diagnostic_trace.json").read_text())
    sources = sorted((output_dir / "videos").rglob("eval_episode_*.mp4"),
                     key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    if len(sources) != len(trace["episodes"]):
        raise RuntimeError("Raw video count differs from diagnostic episode count")
    export_dir = output_dir / "state_changes"
    export_dir.mkdir(exist_ok=True)
    for source, episode in zip(sources, trace["episodes"]):
        index = episode["episode_index"]
        episode_dir = export_dir / f"episode_{index:02}"
        episode_dir.mkdir(exist_ok=True)
        manifest_path = episode_dir / "changes.json"
        if manifest_path.exists():
            prior = json.loads(manifest_path.read_text())
            if (len(prior["changes"]) == len(episode["state_path"])
                    and all((episode_dir / item["image"]).exists() for item in prior["changes"])):
                continue
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {source}")
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count != episode["n_actions"] + 1:
                raise RuntimeError(f"Frame/state mismatch in {source}: {frame_count} frames, "
                                   f"{episode['n_actions']} actions")
            active = set()
            changes = []
            for change in episode["state_path"]:
                step = change["step"]
                active.update(change["added"])
                active.difference_update(change["removed"])
                capture.set(cv2.CAP_PROP_POS_FRAMES, step)
                found, frame = capture.read()
                if not found:
                    raise RuntimeError(f"Cannot read step {step} from {source}")
                image_name = f"step_{step:06}.png"
                if not cv2.imwrite(str(episode_dir / image_name), frame):
                    raise RuntimeError(f"Cannot save {episode_dir / image_name}")
                changes.append({
                    "step": step,
                    "time_seconds": step / fps,
                    "image": image_name,
                    "added": change["added"],
                    "removed": change["removed"],
                    "original_state": sorted(atom for atom in active if not _is_diagnostic(atom)),
                    "diagnostic_state": sorted(atom for atom in active if _is_diagnostic(atom)),
                    "extended_state": sorted(active),
                })
            manifest = {"suite": trace["suite"], "task_id": trace["task_id"],
                        "episode_index": index, "seed": episode["seed"],
                        "success": episode["success"], "fps": fps, "changes": changes}
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        finally:
            capture.release()
    return export_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    export_state_change_frames(parser.parse_args().output_dir)
