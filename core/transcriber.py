import json
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

from config import CACHE_DIR, WHISPER_MODEL


@dataclass
class TranscriptWord:
    start: float
    end: float
    word: str


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    words: list  # list of dicts {start, end, word}


def transcribe(video_path: Path, progress_callback: Optional[Callable] = None) -> list[TranscriptSegment]:
    cache_key = hashlib.md5(str(video_path.resolve()).encode()).hexdigest()
    cache_path = CACHE_DIR / f"transcript_{cache_key}.json"

    if cache_path.exists():
        if progress_callback:
            progress_callback(1.0)
        return _load_cache(cache_path)

    from faster_whisper import WhisperModel

    model = WhisperModel(WHISPER_MODEL, device="auto", compute_type="int8")

    segments_iter, info = model.transcribe(
        str(video_path),
        word_timestamps=True,
        language="en",
    )

    segments: list[TranscriptSegment] = []
    duration = info.duration or 1.0

    for seg in segments_iter:
        words = [{"start": w.start, "end": w.end, "word": w.word} for w in (seg.words or [])]
        segments.append(
            TranscriptSegment(start=seg.start, end=seg.end, text=seg.text.strip(), words=words)
        )
        if progress_callback:
            progress_callback(min(seg.end / duration, 0.99))

    _save_cache(cache_path, segments)

    if progress_callback:
        progress_callback(1.0)

    return segments


def format_for_prompt(segments: list[TranscriptSegment], max_chars: int = 80000) -> str:
    lines = []
    total = 0
    for seg in segments:
        mins = int(seg.start // 60)
        secs = seg.start % 60
        line = f"[{mins:02d}:{secs:05.2f}] {seg.text}"
        total += len(line)
        if total > max_chars:
            lines.append("[transcript truncated for length]")
            break
        lines.append(line)
    return "\n".join(lines)


def get_segments_in_range(
    segments: list[TranscriptSegment], start: float, end: float
) -> list[TranscriptSegment]:
    return [s for s in segments if s.end > start and s.start < end]


def to_srt(segments: list[TranscriptSegment], start_offset: float = 0.0) -> str:
    lines = []
    idx = 1
    for seg in segments:
        t_start = max(seg.start - start_offset, 0)
        t_end = max(seg.end - start_offset, 0)
        if t_end <= 0:
            continue
        lines.append(str(idx))
        lines.append(f"{_srt_time(t_start)} --> {_srt_time(t_end)}")
        lines.append(seg.text)
        lines.append("")
        idx += 1
    return "\n".join(lines)


def _srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _save_cache(path: Path, segments: list[TranscriptSegment]) -> None:
    with open(path, "w") as f:
        json.dump([asdict(s) for s in segments], f)


def _load_cache(path: Path) -> list[TranscriptSegment]:
    with open(path) as f:
        data = json.load(f)
    return [TranscriptSegment(**d) for d in data]
