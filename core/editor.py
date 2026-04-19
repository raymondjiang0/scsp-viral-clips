import subprocess
import tempfile
from pathlib import Path

from config import OUTPUT_DIR


def assemble_clips(clip_paths: list[Path], output_filename: str = "final_edit.mp4") -> Path:
    """Concatenate clips into a single video using ffmpeg concat demuxer."""
    output_path = OUTPUT_DIR / output_filename

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for cp in clip_paths:
            f.write(f"file '{cp.resolve()}'\n")
        concat_list = Path(f.name)

    subprocess.run(
        [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            "-y", str(output_path),
        ],
        check=True,
        capture_output=True,
    )

    concat_list.unlink(missing_ok=True)
    return output_path


def add_text_overlay(
    video_path: Path,
    text: str,
    duration: float = 2.5,
    output_path: Optional[Path] = None,
) -> Path:
    """Burn a full-screen text overlay for the first N seconds (hook intro card)."""
    from typing import Optional

    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_hook.mp4"

    # Escape special chars for drawtext filter
    safe_text = text.replace("'", "\\'").replace(":", "\\:")

    vf = (
        f"drawtext=text='{safe_text}':"
        "fontcolor=white:"
        "fontsize=36:"
        "font=Arial:"
        "fontfile=/System/Library/Fonts/Helvetica.ttc:"
        "x=(w-text_w)/2:"
        "y=(h-text_h)/2:"
        "box=1:"
        "boxcolor=black@0.7:"
        "boxborderw=20:"
        f"enable='between(t,0,{duration})'"
    )

    subprocess.run(
        [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", vf,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "fast",
            "-crf", "20",
            "-y", str(output_path),
        ],
        check=True,
        capture_output=True,
    )
    return output_path
