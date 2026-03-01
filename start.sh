#!/bin/bash
# One-click installer and launcher for Personal Chess AI

echo "🎮 Personal Chess AI - Setup & Launch"
echo "======================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed."
    echo ""
    echo "Please install Python first:"
    echo "  macOS: brew install python@3.11"
    echo "  Linux: sudo apt install python3"
    echo "  Windows: Download from https://www.python.org/downloads/"
    echo ""
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Check/Install Stockfish
if ! command -v stockfish &> /dev/null; then
    echo "⚙️  Stockfish not found. Installing..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            echo "Installing Stockfish via Homebrew..."
            brew install stockfish
        else
            echo "❌ Homebrew not found. Please install Stockfish manually:"
            echo "   brew install stockfish"
            echo "   Or download from: https://stockfishchess.org/download/"
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo "Installing Stockfish..."
        sudo apt update && sudo apt install -y stockfish
    else
        echo "❌ Please install Stockfish manually:"
        echo "   Download from: https://stockfishchess.org/download/"
        exit 1
    fi
    
    if command -v stockfish &> /dev/null; then
        echo "✅ Stockfish installed successfully!"
    else
        echo "❌ Stockfish installation failed. Please install manually."
        exit 1
    fi
else
    echo "✅ Stockfish found"
fi

# Setup Python environment
if [ ! -d "venv" ]; then
    echo "📦 Setting up Python environment..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Create saved_games folder
mkdir -p saved_games

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 Starting Chess AI..."
echo ""

# Open browser automatically after 2 seconds
(sleep 2 && python3 -m webbrowser http://localhost:8080) &

# Start the app
python app.py
