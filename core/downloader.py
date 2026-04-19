import json
import subprocess
from pathlib import Path
from typing import Callable, Optional

import yt_dlp

from config import DATA_DIR


def download_youtube(url: str, progress_callback: Optional[Callable] = None) -> Path:
    downloaded_path = [None]

    def progress_hook(d):
        if d["status"] == "finished":
            downloaded_path[0] = Path(d["filename"])
        if d["status"] == "downloading" and progress_callback:
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 1
            done = d.get("downloaded_bytes", 0)
            progress_callback(done / total)

    ydl_opts = {
        # ios/android clients bypass YouTube's SABR streaming 403 errors
        "extractor_args": {"youtube": {"player_client": ["ios", "android", "tv_embedded"]}},
        "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best[ext=mp4]/best",
        "outtmpl": str(DATA_DIR / "%(title)s.%(ext)s"),
        "progress_hooks": [progress_hook],
        "merge_output_format": "mp4",
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    path = Path(filename)
    if not path.exists():
        # Try with .mp4 extension
        path = path.with_suffix(".mp4")
    if not path.exists() and downloaded_path[0]:
        path = downloaded_path[0]

    return path


def save_upload(uploaded_file) -> Path:
    output_path = DATA_DIR / uploaded_file.name
    with open(output_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return output_path


def get_video_duration(video_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(video_path)],
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            return float(stream.get("duration", 0))
    # Fallback: check audio stream
    for stream in data.get("streams", []):
        duration = stream.get("duration")
        if duration:
            return float(duration)
    return 0.0


def preprocess_for_gemini(video_path: Path) -> Path:
    """Downsample video if over ~1.8GB to stay within Gemini File API limits."""
    import os

    size_mb = os.path.getsize(video_path) / (1024 * 1024)
    if size_mb < 1800:
        return video_path

    output_path = DATA_DIR / f"{video_path.stem}_gemini.mp4"
    if output_path.exists():
        return output_path

    subprocess.run(
        [
            "ffmpeg", "-i", str(video_path),
            "-vf", "scale=720:-2",
            "-b:v", "1M",
            "-b:a", "128k",
            "-y", str(output_path),
        ],
        check=True,
        capture_output=True,
    )
    return output_path


def check_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
