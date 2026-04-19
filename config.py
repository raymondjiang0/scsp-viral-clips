import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / ".cache"
RAG_DB_DIR = BASE_DIR / "rag_db"
OUTPUT_DIR = BASE_DIR / "outputs"

for _d in [DATA_DIR, CACHE_DIR, RAG_DB_DIR, OUTPUT_DIR]:
    _d.mkdir(exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

BRAND_GUIDE_PATH = Path(os.getenv(
    "BRAND_GUIDE_PATH",
    str(BASE_DIR.parent / "PTB Brand Guide.pdf")
))
SCSP_WEBSITE_URL = "https://www.scsp.ai"

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")

MAX_INSTAGRAM_DURATION = 60
MAX_LINKEDIN_DURATION = 90
CLIP_BUFFER_SECONDS = 1.5

# Caption style (burned into clips)
CAPTION_FONTSIZE = 22
CAPTION_FONT = "Arial"
CAPTION_FONTCOLOR = "white"
CAPTION_BOX_COLOR = "0x000000@0.6"
CAPTION_MARGIN_V = 35
