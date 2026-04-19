import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from config import CACHE_DIR, CLIP_BUFFER_SECONDS, OUTPUT_DIR
from core.transcriber import TranscriptSegment, get_segments_in_range, to_srt


def extract_clip(
    video_path: Path,
    start: float,
    end: float,
    clip_id: str,
    segments: list[TranscriptSegment],
    burn_captions: bool = True,
) -> Path:
    output_path = OUTPUT_DIR / f"{clip_id}.mp4"
    if output_path.exists():
        return output_path

    # Add buffer but don't go below 0
    t_start = max(start - CLIP_BUFFER_SECONDS, 0)
    t_end = end + CLIP_BUFFER_SECONDS
    duration = t_end - t_start

    if not burn_captions or not segments:
        _extract_raw(video_path, t_start, duration, output_path)
        return output_path

    # Extract raw clip first, then burn captions
    raw_path = OUTPUT_DIR / f"{clip_id}_raw.mp4"
    _extract_raw(video_path, t_start, duration, raw_path)

    # Build SRT for this clip's time range
    clip_segments = get_segments_in_range(segments, start, end)
    srt_content = to_srt(clip_segments, start_offset=t_start)

    if not srt_content.strip():
        raw_path.rename(output_path)
        return output_path

    srt_path = OUTPUT_DIR / f"{clip_id}.srt"
    srt_path.write_text(srt_content, encoding="utf-8")

    _burn_subtitles(raw_path, srt_path, output_path)

    # Cleanup intermediates
    raw_path.unlink(missing_ok=True)
    srt_path.unlink(missing_ok=True)

    return output_path


def extract_thumbnail(video_path: Path, timestamp: float, clip_id: str) -> Path:
    thumb_path = CACHE_DIR / f"thumb_{clip_id}.jpg"
    if thumb_path.exists():
        return thumb_path

    subprocess.run(
        [
            "ffmpeg", "-ss", str(timestamp),
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "3",
            "-y", str(thumb_path),
        ],
        capture_output=True,
        check=True,
    )
    return thumb_path


def _extract_raw(video_path: Path, start: float, duration: float, output_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-ss", str(start),
            "-i", str(video_path),
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "fast",
            "-crf", "20",
            "-y", str(output_path),
        ],
        check=True,
        capture_output=True,
    )


def _burn_subtitles(video_path: Path, srt_path: Path, output_path: Path) -> None:
    # Escape path for ffmpeg filter — colons and backslashes need escaping
    srt_escaped = str(srt_path).replace("\\", "/").replace(":", "\\:")

    # Modern-looking captions: white bold text, semi-transparent black box
    style = (
        "FontName=Arial,"
        "FontSize=22,"
        "Bold=1,"
        "PrimaryColour=&HFFFFFF,"
        "OutlineColour=&H00000000,"
        "BackColour=&H99000000,"
        "BorderStyle=3,"
        "Outline=0,"
        "Shadow=0,"
        "Alignment=2,"
        "MarginV=35"
    )

    vf = f"subtitles='{srt_escaped}':force_style='{style}'"

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
