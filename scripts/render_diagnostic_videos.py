"""Burn frame-aligned BDDL and diagnostic captions into LIBERO videos."""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg


DIAGNOSTIC_PREDICATES = {
    "initial_orientation", "fallen", "holding", "on_center", "in_center",
    "closed", "partially_open", "fully_open",
}


def _ass_time(seconds):
    centiseconds = round(seconds * 100)
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    whole, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02}:{whole:02}.{fraction:02}"


def _ass_text(value):
    return value.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _dialogue(start, end, style, lines):
    text = r"\N".join(_ass_text(line) for line in lines)
    return f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},{style},,0,0,0,,{text}\n"


def _state_events(episode, fps, frame_count):
    active = set()
    events = []
    path = episode["state_path"]
    for index, change in enumerate(path):
        active.update(change["added"])
        active.difference_update(change["removed"])
        start = change["step"] / fps
        stop_step = path[index + 1]["step"] if index + 1 < len(path) else frame_count
        end = stop_step / fps
        if end <= start:
            continue
        original = sorted(atom for atom in active if atom[1:].split(" ", 1)[0]
                          not in DIAGNOSTIC_PREDICATES)
        extended = sorted(active)
        events.append(_dialogue(start, end, "Original", original or ["(none)"]))
        events.append(_dialogue(start, end, "Extended", extended or ["(none)"]))
    return "".join(events)


def _caption_font_size(episode):
    active = set()
    longest = 1
    most_lines = 1
    for change in episode["state_path"]:
        active.update(change["added"])
        active.difference_update(change["removed"])
        longest = max(longest, max((len(atom) for atom in active), default=0))
        original_count = sum(atom[1:].split(" ", 1)[0] not in DIAGNOSTIC_PREDICATES
                             for atom in active)
        most_lines = max(most_lines, original_count, len(active))
    return max(9, min(18, int(340 / (1.2 * most_lines)), int(480 / (0.62 * longest))))


def _video_info(path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        return capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def annotate_task_videos(output_dir, overwrite=False):
    output_dir = Path(output_dir)
    trace = json.loads((output_dir / "diagnostic_trace.json").read_text())
    video_dir = output_dir / "annotated_videos"
    video_dir.mkdir(exist_ok=True)
    raw_videos = sorted((output_dir / "videos").rglob("eval_episode_*.mp4"),
                        key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    if len(raw_videos) != len(trace["episodes"]):
        raise RuntimeError("Raw video count differs from diagnostic episode count")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for source, episode in zip(raw_videos, trace["episodes"]):
        index = episode["episode_index"]
        fps, frame_count = _video_info(source)
        if frame_count != episode["n_actions"] + 1:
            raise RuntimeError(f"Frame/state mismatch in {source}: {frame_count} frames, "
                               f"{episode['n_actions']} actions")
        destination = video_dir / f"episode_{index:02}.mp4"
        if destination.exists() and not overwrite:
            continue
        duration = frame_count / fps
        font_size = _caption_font_size(episode)
        caption = f"{trace['suite']} task {trace['task_id']:02}  |  episode {index:02}  |  seed {episode['seed']}"
        ass = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1024
PlayResY: 1024
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Original,DejaVu Sans Mono,{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,12,0,657,1
Style: Extended,DejaVu Sans Mono,{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,524,0,657,1
Style: Header,DejaVu Sans,22,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,0,0,7,12,0,618,1
Style: RightHeader,DejaVu Sans,22,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,0,0,7,524,0,618,1
Style: Footer,DejaVu Sans,20,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,8,0,0,12,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        ass += _dialogue(0, duration, "Header", ["Original task BDDL"])
        ass += _dialogue(0, duration, "RightHeader", ["Extended state"])
        ass += _dialogue(0, duration, "Footer", [caption])
        ass += _state_events(episode, fps, frame_count)
        subtitle_path = video_dir / f"episode_{index:02}.ass"
        subtitle_path.write_text(ass)
        temporary = destination.with_suffix(".tmp.mp4")
        filter_graph = ("scale=1024:1024:flags=lanczos,"
                        "drawbox=x=0:y=0:w=1024:h=38:color=0x101820@0.7:t=fill,"
                        "drawbox=x=0:y=602:w=1024:h=422:color=0x101820@0.78:t=fill,"
                        "drawbox=x=511:y=602:w=2:h=422:color=0xffffff@0.35:t=fill,"
                        f"ass={subtitle_path}")
        try:
            subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                            "-vf", filter_graph, "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "25", "-pix_fmt", "yuv420p", "-an", str(temporary)], check=True)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
            subtitle_path.unlink(missing_ok=True)
    return video_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    annotate_task_videos(args.output_dir, overwrite=args.overwrite)
