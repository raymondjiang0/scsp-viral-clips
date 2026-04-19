#!/bin/bash
set -e

echo "=== SCSP Viral Clip Engine — Setup ==="

# Add Homebrew to PATH for Apple Silicon and Intel Macs
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

# Check for ffmpeg — install Homebrew first if needed
if ! command -v ffmpeg &> /dev/null; then
    if ! command -v brew &> /dev/null; then
        echo "Installing Homebrew (required for ffmpeg)..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        # Re-source Homebrew after install
        export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
        eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || /usr/local/bin/brew shellenv 2>/dev/null || true)"
    fi
    echo "Installing ffmpeg..."
    brew install ffmpeg
else
    echo "✅ ffmpeg found"
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing Python dependencies (this may take a few minutes)..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Copy env file
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "⚠️  Created .env — please add your GEMINI_API_KEY:"
    echo "    nano .env"
else
    echo "✅ .env exists"
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To run the app:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
echo ""
echo "To share with a coworker, they just need to:"
echo "  1. Clone/copy this folder"
echo "  2. Run: bash setup.sh"
echo "  3. Add their GEMINI_API_KEY to .env"
echo "  4. Run: source venv/bin/activate && streamlit run app.py"
