@echo off
REM One-click launcher for Personal Chess AI (Windows)

echo ========================================
echo   Personal Chess AI - Setup ^& Launch
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed.
    echo.
    echo Please install Python first:
    echo   Download from: https://www.python.org/downloads/
    echo   Make sure to check "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

echo [OK] Python found
echo.

REM Check Stockfish
where stockfish >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Stockfish not found.
    echo.
    echo Please install Stockfish:
    echo   1. Download from: https://stockfishchess.org/download/windows/
    echo   2. Extract to C:\Program Files\Stockfish\
    echo   3. Add to PATH or update stockfish_analyzer.py with the path
    echo.
    echo Press any key to continue anyway...
    pause >nul
) else (
    echo [OK] Stockfish found
    echo.
)

REM Setup virtual environment
if not exist "venv" (
    echo Setting up Python environment...
    python -m venv venv
)

call venv\Scripts\activate.bat
echo Installing dependencies...
pip install -q --upgrade pip
pip install -q -r requirements.txt

REM Create saved_games folder
if not exist "saved_games" mkdir saved_games

echo.
echo [OK] Setup complete!
echo.
echo Starting Chess AI...
echo.

REM Open browser after 3 seconds
start /B timeout /t 3 /nobreak >nul && start http://localhost:8080

REM Start the app
python app.py
