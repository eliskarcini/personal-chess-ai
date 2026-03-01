from flask import Flask, render_template, request, jsonify, session
import uuid
import time
from chess_engine import ChessEngine
import chess
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

# Global chess engines for each session
engines = {}

def get_engine(session_id):
    """Get or create chess engine for session"""
    if session_id not in engines:
        engines[session_id] = ChessEngine(
            basic_model_path="models/best_model.pth",
            hybrid_model_path="models/hybrid_chess_model.pt",
            pgn_data_path=None
        )
    return engines[session_id]

@app.route('/')
def index():
    """Main chess game page"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('chess.html')

@app.route('/new_game', methods=['POST'])
def new_game():
    """Start a new chess game"""
    data = request.get_json()
    player_side = data.get('player_side', 'white')  # 'white' or 'black'
    player_name = data.get('player_name', 'Human')
    
    session_id = session.get('session_id')
    engine = get_engine(session_id)
    engine.reset_game()
    
    # Store game settings
    engine.player_side = player_side
    engine.player_name = player_name
    engine.game_start_time = time.time()
    
    result = {
        'success': True,
        'board': str(engine.board),
        'fen': engine.board.fen(),
        'legal_moves': engine.get_legal_moves(),
        'game_state': engine._get_game_state(),
        'player_side': player_side
    }
    
    # If player chose black, AI makes first move
    if player_side == 'black':
        ai_result = engine.get_ai_move(player_rating=1500, use_hybrid=True)
        if ai_result['success']:
            result.update({
                'ai_first_move': ai_result['ai_move'],
                'board': ai_result['move_result']['board'],
                'fen': ai_result['move_result']['fen'],
                'legal_moves': engine.get_legal_moves(),
                'game_state': ai_result['move_result']['game_state']
            })
    
    return jsonify(result)

@app.route('/make_move', methods=['POST'])
def make_move():
    """Make a human move"""
    data = request.get_json()
    move = data.get('move')
    player_rating = data.get('rating', 1500)
    use_hybrid = data.get('use_hybrid', True)
    strength_level = data.get('strength_level', 1500)
    
    session_id = session.get('session_id')
    engine = get_engine(session_id)
    
    result = engine.make_move(move)
    
    if result['success']:
        result['legal_moves'] = engine.get_legal_moves()
        
        # Check if game is over
        if engine.board.is_game_over():
            # Auto-save completed game
            auto_save_result = auto_save_game(engine)
            result['auto_saved'] = auto_save_result
        else:
            # Auto-play AI response with strength level
            ai_result = engine.get_ai_move(
                player_rating=player_rating,
                use_hybrid=use_hybrid,
                thinking_time=1.0,
                strength_level=strength_level
            )
            
            if ai_result['success']:
                result['ai_response'] = {
                    'move': ai_result['ai_move'],
                    'thinking_time': ai_result['thinking_time'],
                    'analysis': ai_result['analysis'],
                    'board': ai_result['move_result']['board'],
                    'fen': ai_result['move_result']['fen'],
                    'game_state': ai_result['move_result']['game_state']
                }
                result['legal_moves'] = engine.get_legal_moves()
                
                # Check if game ended after AI move
                if engine.board.is_game_over():
                    auto_save_result = auto_save_game(engine)
                    result['auto_saved'] = auto_save_result
    
    return jsonify(result)

@app.route('/ai_move', methods=['POST'])
def ai_move():
    """Get AI move"""
    data = request.get_json()
    player_rating = data.get('rating', 1500)
    use_hybrid = data.get('use_hybrid', True)
    
    session_id = session.get('session_id')
    engine = get_engine(session_id)
    
    result = engine.get_ai_move(
        player_rating=player_rating,
        use_hybrid=use_hybrid,
        thinking_time=1.0
    )
    
    if result['success']:
        result['legal_moves'] = engine.get_legal_moves()
    
    return jsonify(result)

@app.route('/analyze_position', methods=['POST'])
def analyze_position():
    """Get position analysis"""
    data = request.get_json()
    player_rating = data.get('rating', 1500)
    
    session_id = session.get('session_id')
    engine = get_engine(session_id)
    
    analysis = engine.get_position_analysis(player_rating)
    return jsonify(analysis)

@app.route('/save_game', methods=['POST'])
def save_game():
    """Save current game as PGN"""
    data = request.get_json()
    filename = data.get('filename', 'game.pgn')
    
    session_id = session.get('session_id')
    engine = get_engine(session_id)
    
    # Create games directory if it doesn't exist
    os.makedirs('saved_games', exist_ok=True)
    filepath = os.path.join('saved_games', filename)
    
    engine.save_game(filepath)
    
    return jsonify({
        'success': True,
        'message': f'Game saved as {filename}',
        'filepath': filepath
    })

def auto_save_game(engine):
    """Automatically save completed game"""
    try:
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Determine players based on side choice
        if hasattr(engine, 'player_side') and engine.player_side == 'black':
            white_player = "AI"
            black_player = getattr(engine, 'player_name', 'Human')
        else:
            white_player = getattr(engine, 'player_name', 'Human')
            black_player = "AI"
        
        filename = f"game_{timestamp}_{white_player}_vs_{black_player}.pgn"
        
        os.makedirs('saved_games', exist_ok=True)
        filepath = os.path.join('saved_games', filename)
        
        engine.save_game(filepath, white_player=white_player, black_player=black_player)
        
        return {
            'success': True,
            'message': f'Game automatically saved as {filename}',
            'filepath': filepath
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Auto-save failed: {str(e)}'
        }
    """Get list of saved games"""
    games_dir = 'saved_games'
    if not os.path.exists(games_dir):
        return jsonify({'games': []})
    
    games = []
    for filename in os.listdir(games_dir):
        if filename.endswith('.pgn'):
            filepath = os.path.join(games_dir, filename)
            stat = os.stat(filepath)
            games.append({
                'filename': filename,
                'size': stat.st_size,
                'modified': stat.st_mtime
            })
    
    # Sort by modification time (newest first)
    games.sort(key=lambda x: x['modified'], reverse=True)
    
    return jsonify({'games': games})

if __name__ == '__main__':
    print("🚀 Starting Chess AI Web App...")
    print("📱 Open http://localhost:8080 in your browser")
    print("🤖 AI models loaded and ready!")
    app.run(debug=True, host='0.0.0.0', port=8080)
