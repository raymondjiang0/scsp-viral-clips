# SCSP Viral Clip Engine

Ingests long-form video (YouTube or MP4) and outputs viral short-form clips for Instagram and LinkedIn, tailored to the SCSP brand and audience.

## What it does

1. Downloads a YouTube VOD or accepts an uploaded MP4
2. Transcribes the audio with Whisper
3. Sends video + transcript to Gemini 2.5 Flash for viral clip analysis
4. Extracts suggested clips with captions burned in
5. Assembles a finished edit with hook overlay
6. Generates original content ideas to film separately

Uses a RAG system built on the PTB Brand Guide and SCSP website to score clips against brand voice.

## Prerequisites

- Python 3.10+ recommended (3.9 works with warnings)
- A Gemini API key — get one at [aistudio.google.com](https://aistudio.google.com)
- ffmpeg (installed automatically by `setup.sh` via Homebrew on Mac)

## Setup

```bash
git clone <repo-url>
cd scsp-viral-clips
bash setup.sh
```

`setup.sh` will:
- Install Homebrew if needed
- Install ffmpeg
- Create a Python virtual environment
- Install all dependencies
- Create a `.env` file from `.env.example`

Then add your API key to `.env`:

```
GEMINI_API_KEY=your_key_here
```

## Running

```bash
source venv/bin/activate
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Brand Guide

Place `PTB Brand Guide.pdf` in the parent directory (one level above this repo), or set the path in `.env`:

```
BRAND_GUIDE_PATH=/path/to/PTB Brand Guide.pdf
```

On first run, click **"(Re)index Brand Guide + Website"** in the sidebar to build the knowledge base. This takes ~1-2 minutes and only needs to be done once (or after the brand guide is updated).

## Usage

### Step 1 — Process Video tab
- Paste a YouTube URL or upload an MP4
- Click **Download Video** (YouTube) or **Use this file** (upload)
- Click **Analyze for Viral Clips** — runs the full pipeline:
  - Whisper transcribes audio
  - Gemini analyzes for viral moments
  - Clips extracted with captions burned in

Processing time: ~5-15 min depending on video length.

### Step 2 — Instagram Clips / LinkedIn Clips tabs
Each clip shows: virality score, hook, why it goes viral, suggested caption + hashtags, timestamp range, preview, and download.

Sort by virality score. Use **Add to final edit** to queue clips for the finished product.

### Step 3 — Finished Edit tab
Gemini pre-selects the best clips in narrative order. Review, add/remove clips, optionally add a hook text overlay, then click **Generate Final Video** to export.

### Step 4 — Content Ideas tab
5 original video concepts to film from scratch, each with hook, outline, and trending angle. Save ideas to `outputs/content_calendar.txt`.

## Whisper model quality

Set `WHISPER_MODEL` in `.env`:

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `base` | 74 MB | Fast | Good for testing |
| `small` | 244 MB | Medium | Good for production |
| `large-v3` | 3 GB | Slow | Best accuracy |

## Sharing with coworkers

Each person needs to:
1. Clone the repo
2. Run `bash setup.sh`
3. Add their own `GEMINI_API_KEY` to `.env`
4. Run `source venv/bin/activate && streamlit run app.py`

For a shared hosted version, deploy to [Streamlit Community Cloud](https://streamlit.io/cloud) (free).
