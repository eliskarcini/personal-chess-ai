# ♟️ Personal Chess AI

Play chess against an AI that learns and adapts to your playing style.

## Features

- **Hybrid AI**: Combines Stockfish engine with neural network for human-like play
- **Adaptive Difficulty**: Adjustable from 800 to 2000+ ELO
- **Style Learning**: AI learns from your games
- **Web Interface**: Play in your browser
- **Auto-Save**: All games saved in PGN format

## Quick Start

### For Non-Technical Users (Easiest Way)

**Step 1:** Install Python
- Download from https://www.python.org/downloads/
- Windows users: Check "Add Python to PATH" during installation

**Step 2:** Download this project
- Click the green **"Code"** button → **"Download ZIP"**
- Extract the ZIP file

**Step 3:** Run the launcher

**macOS:**
1. Right-click the `personal-chess-ai` folder
2. Select "New Terminal at Folder"
3. Type: `./start.sh` and press Enter

**Linux:**
1. Open terminal in the project folder
2. Type: `./start.sh` and press Enter

**Windows:**
- Double-click `start.bat`

The launcher auto-installs everything and opens your browser!

**Windows users only:** Manually install Stockfish from https://stockfishchess.org/download/

### For Developers

**Requirements:**
- Python 3.8+
- Stockfish chess engine

**Installation:**

**macOS:**
```bash
# Install Stockfish
brew install stockfish

# Run the game
./run.sh
```

**Linux:**
```bash
# Install Stockfish
sudo apt install stockfish

# Run the game
./run.sh
```

**Windows:**
```bash
# Install Stockfish from https://stockfishchess.org/download/
# Then run:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### Play!

Open your browser: **http://localhost:8080**

## How It Works

The AI uses a hybrid approach:
- **80% Stockfish**: Strong tactical play
- **20% Neural Network**: Human-like style and personality

The neural network is trained on my chess games and at the time imitates my game style.

## Project Structure

```
personal-chess-ai/
├── app.py                 # Flask web server
├── chess_engine.py        # Game logic
├── enhanced_ai.py         # Hybrid AI
├── stockfish_analyzer.py  # Stockfish interface
├── model_interface.py     # Neural network interface
├── templates/             # Web UI
├── models/                # Pre-trained models
└── saved_games/           # Your games (auto-created)
```

## Usage

1. Start the server with `./start.sh` (or `start.bat` on Windows)
2. Browser opens automatically to http://localhost:8080
3. Click pieces to move them
4. Adjust AI difficulty with the slider (800-2000 ELO)
5. **Games automatically save** to `saved_games/` folder when finished

Every completed game is saved as a PGN file with a timestamp - perfect for analyzing your progress!

## 🔮 What's Next

This project is actively being developed! Upcoming features:

- **🎯 Personal Style Learning**: Retrain the AI on your saved games to learn YOUR unique playing style
- **📊 Adaptive Opponents**: AI that adapts to your strengths and weaknesses over time
- **📚 Opening Book**: Learn and play your favorite openings
- **🔍 Game Analysis**: Post-game analysis showing mistakes and suggested improvements

## License

MIT License - Free to use and modify


