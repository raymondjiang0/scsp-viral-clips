import json
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL, CACHE_DIR

_ANALYSIS_PROMPT = """You are a viral content strategist for SCSP (Special Competitive Studies Project), a nonpartisan think tank dedicated to ensuring the United States wins the technology competition — especially in AI — against strategic rivals.

=== BRAND & AUDIENCE CONTEXT ===
{brand_context}

=== TARGET AUDIENCES ===
Instagram: Gen Z interested in defense, AI, and geopolitics. Wants punchy, fast, surprising content. Responds to urgency and "you didn't know this" energy.
LinkedIn: Defense officials, national security professionals, DC policy crowd, tech leaders, government contractors. Wants credible insight, bold takes, thought leadership.

=== WHAT GOES VIRAL FOR SCSP ===
Instagram triggers: "China is already doing X", counter-intuitive facts about US military/AI, scary-but-true statistics, behind-the-scenes of how national security actually works, "here's what no one is talking about" energy. Keep it punchy — no jargon, no hedging.
LinkedIn triggers: Bold expert predictions, data-backed arguments, policy implications that affect careers/industries, insider perspective on defense/AI that outsiders don't get.

=== VIDEO TRANSCRIPT (with timestamps) ===
{transcript}

=== TASK ===
Identify the best moments for viral short-form clips. Prioritize:
1. Counter-intuitive or surprising statements
2. Specific numbers, statistics, comparisons
3. Provocative predictions or opinions stated with conviction
4. Moments of genuine emotion or urgency
5. Clear "aha moment" explanations of complex topics
6. Any direct China/AI/US competition references

IMPORTANT: Use the transcript timestamps to pinpoint exact moments.

Return ONLY a valid JSON object — no markdown fences, no explanation text:
{{
  "instagram_clips": [
    {{
      "clip_id": "ig_1",
      "start_seconds": 142.5,
      "end_seconds": 178.0,
      "virality_score": 8.7,
      "hook": "Exact first sentence spoken in this clip",
      "why_viral": "Specific explanation of why this lands for Instagram/Gen Z SCSP audience",
      "content_type": "surprising_fact|expert_insight|provocative_claim|emotional_moment|data_stat",
      "suggested_caption": "Full Instagram caption with line breaks and emojis",
      "suggested_hashtags": ["#NationalSecurity", "#AI", "#SCSP", "#DefenseTech"]
    }}
  ],
  "linkedin_clips": [
    {{
      "clip_id": "li_1",
      "start_seconds": 312.0,
      "end_seconds": 395.0,
      "virality_score": 9.1,
      "hook": "Exact first sentence spoken in this clip",
      "why_viral": "Specific explanation for LinkedIn/professional SCSP audience",
      "content_type": "expert_insight",
      "suggested_caption": "Full LinkedIn caption with professional framing",
      "suggested_hashtags": ["#NationalSecurity", "#AIPolicy", "#Technology"]
    }}
  ],
  "finished_product": {{
    "clips_in_order": ["ig_1", "ig_2", "ig_3"],
    "platform": "instagram",
    "narrative_arc": "How these clips together build a compelling narrative",
    "suggested_hook_text": "Bold 2-second text overlay for opening frame",
    "suggested_cta": "End screen call-to-action text"
  }},
  "content_suggestions": [
    {{
      "title": "Specific video concept title",
      "format": "talking_head|screen_recording|interview_clip|b_roll_voiceover|trending_format",
      "hook": "Exact words/visuals for the opening 2 seconds",
      "outline": ["Point 1 with specifics", "Point 2 with specifics", "Point 3 with specifics"],
      "why_viral": "Why this would go viral specifically for SCSP",
      "target_platform": "instagram|linkedin|both",
      "trending_angle": "Specific current trend, news hook, or cultural moment this taps into"
    }}
  ]
}}

Provide 4-5 Instagram clips, 4-5 LinkedIn clips, and 5 content suggestions. Be specific, opinionated, and direct."""


@dataclass
class ClipSuggestion:
    clip_id: str
    start_seconds: float
    end_seconds: float
    virality_score: float
    hook: str
    why_viral: str
    content_type: str
    suggested_caption: str
    suggested_hashtags: list
    platform: str = ""

    @property
    def duration(self):
        return self.end_seconds - self.start_seconds


@dataclass
class ContentSuggestion:
    title: str
    format: str
    hook: str
    outline: list
    why_viral: str
    target_platform: str
    trending_angle: str


@dataclass
class FinishedProduct:
    clips_in_order: list
    platform: str
    narrative_arc: str
    suggested_hook_text: str
    suggested_cta: str = ""


@dataclass
class AnalysisResult:
    instagram_clips: list
    linkedin_clips: list
    finished_product: FinishedProduct
    content_suggestions: list


def analyze_video(
    video_path: Path,
    brand_context: str,
    transcript_text: str,
    progress_callback: Optional[Callable] = None,
) -> AnalysisResult:
    cache_key = hashlib.md5(f"{video_path.resolve()}{brand_context[:100]}".encode()).hexdigest()
    cache_path = CACHE_DIR / f"analysis_{cache_key}.json"

    if cache_path.exists():
        if progress_callback:
            progress_callback(1.0, "Loaded from cache")
        return _load_cache(cache_path)

    client = genai.Client(api_key=GEMINI_API_KEY)

    if progress_callback:
        progress_callback(0.05, "Uploading video to Gemini…")

    uploaded = client.files.upload(file=str(video_path))

    while uploaded.state.name == "PROCESSING":
        time.sleep(5)
        uploaded = client.files.get(name=uploaded.name)
        if progress_callback:
            progress_callback(0.3, "Gemini processing video…")

    if uploaded.state.name != "ACTIVE":
        raise RuntimeError(f"Gemini video processing failed: {uploaded.state.name}")

    if progress_callback:
        progress_callback(0.55, "Analyzing clips for virality…")

    prompt = _ANALYSIS_PROMPT.format(
        brand_context=brand_context,
        transcript=transcript_text,
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[uploaded, prompt],
        config=types.GenerateContentConfig(temperature=0.7),
    )

    if progress_callback:
        progress_callback(0.9, "Parsing results…")

    raw_text = response.text
    result = _parse_response(raw_text)

    with open(cache_path, "w") as f:
        json.dump(raw_text, f)

    try:
        client.files.delete(name=uploaded.name)
    except Exception:
        pass

    if progress_callback:
        progress_callback(1.0, "Analysis complete!")

    return result


def _parse_response(text: str) -> AnalysisResult:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]

    data = json.loads(text)

    instagram_clips = [
        ClipSuggestion(platform="instagram", **c) for c in data.get("instagram_clips", [])
    ]
    linkedin_clips = [
        ClipSuggestion(platform="linkedin", **c) for c in data.get("linkedin_clips", [])
    ]

    fp = data.get("finished_product", {})
    finished_product = FinishedProduct(
        clips_in_order=fp.get("clips_in_order", []),
        platform=fp.get("platform", "instagram"),
        narrative_arc=fp.get("narrative_arc", ""),
        suggested_hook_text=fp.get("suggested_hook_text", ""),
        suggested_cta=fp.get("suggested_cta", ""),
    )

    content_suggestions = [ContentSuggestion(**s) for s in data.get("content_suggestions", [])]

    return AnalysisResult(
        instagram_clips=instagram_clips,
        linkedin_clips=linkedin_clips,
        finished_product=finished_product,
        content_suggestions=content_suggestions,
    )


def _load_cache(path: Path) -> AnalysisResult:
    with open(path) as f:
        text = json.load(f)
    return _parse_response(text)
