#!/bin/bash
# Simple run script for macOS/Linux users

echo "🎮 Personal Chess AI"
echo "===================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Install from https://www.python.org"
    exit 1
fi

# Check Stockfish
if ! command -v stockfish &> /dev/null; then
    echo "⚠️  Stockfish not found."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Install with: brew install stockfish"
    else
        echo "Install with: sudo apt install stockfish"
    fi
    exit 1
fi

# Setup venv
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate and install
source venv/bin/activate
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Create saved_games folder
mkdir -p saved_games

# Run
echo ""
echo "✅ Starting server..."
echo "🌐 Open: http://localhost:8080"
echo ""
python app.py
