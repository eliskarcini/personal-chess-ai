import torch
import chess
import chess.pgn
from src.hybrid_model import HybridChessModel, ChessEngineIntegration
import numpy as np

class HybridPredictor:
    def __init__(self, model_path, engine_path="/usr/games/stockfish"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = HybridChessModel(vocab_size=4096, rating_bins=10)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize engine
        try:
            self.engine = ChessEngineIntegration(engine_path=engine_path, depth=10)
            print("Engine loaded successfully")
        except:
            print("Warning: Engine not available")
            self.engine = None
    
    def predict_next_move(self, game_moves, player_rating, show_analysis=True):
        """Predict next move with hybrid approach"""
        
        # Convert moves to board position
        board = chess.Board()
        for move in game_moves:
            board.push_san(move)
        
        # Get engine analysis
        engine_features = np.zeros(5)
        engine_suggestions = []
        
        if self.engine:
            engine_moves = self.engine.get_top_moves(board.fen(), top_k=5)
            engine_suggestions = [m['move'] for m in engine_moves]
            
            if engine_moves:
                # Create engine features (simplified for prediction)
                engine_features[0] = engine_moves[0]['centipawn'] / 100.0
                engine_features[3] = len(engine_moves) / 5.0
        
        # Prepare model input
        rating_bin = min(9, max(0, (player_rating - 800) // 200))
        
        # Convert moves to tokens (simplified - you'd need proper tokenization)
        move_tokens = [hash(move) % 4096 for move in game_moves[-10:]]  # Last 10 moves
        
        with torch.no_grad():
            sequence = torch.tensor([move_tokens], dtype=torch.long).to(self.device)
            rating_tensor = torch.tensor([rating_bin], dtype=torch.long).to(self.device)
            engine_tensor = torch.tensor([engine_features], dtype=torch.float32).to(self.device)
            
            outputs = self.model(sequence, rating_tensor, engine_tensor)
            
            # Get predictions
            hybrid_probs = torch.softmax(outputs['hybrid_logits'], dim=-1)
            style_probs = torch.softmax(outputs['style_logits'], dim=-1)
            engine_probs = torch.softmax(outputs['engine_logits'], dim=-1)
            balance_weight = outputs['balance_weight'].item()
        
        if show_analysis:
            print(f"\n=== HYBRID CHESS ANALYSIS ===")
            print(f"Position: {board.fen()}")
            print(f"Player Rating: {player_rating} (Bin: {rating_bin})")
            print(f"Balance Weight: {balance_weight:.3f} (Style vs Engine)")
            
            if engine_suggestions:
                print(f"\nEngine Top Moves:")
                for i, move in enumerate(engine_suggestions[:3]):
                    print(f"  {i+1}. {move}")
            
            print(f"\nModel Analysis:")
            print(f"  Style Component Weight: {balance_weight:.1%}")
            print(f"  Engine Component Weight: {(1-balance_weight):.1%}")
            
            # Get legal moves and their probabilities
            legal_moves = list(board.legal_moves)
            move_probs = []
            
            for move in legal_moves[:5]:  # Top 5 legal moves
                move_token = hash(str(move)) % 4096
                prob = hybrid_probs[0, move_token].item()
                move_probs.append((str(move), prob))
            
            move_probs.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\nHybrid Predictions:")
            for i, (move, prob) in enumerate(move_probs[:3]):
                engine_rank = "Engine Top 3" if move in engine_suggestions[:3] else "Not in Engine Top 3"
                print(f"  {i+1}. {move} ({prob:.1%}) - {engine_rank}")
        
        return {
            'balance_weight': balance_weight,
            'engine_suggestions': engine_suggestions,
            'hybrid_predictions': move_probs,
            'position_fen': board.fen()
        }

def analyze_game_style(pgn_file, player_rating, model_path):
    """Analyze a complete game showing style vs engine preferences"""
    
    predictor = HybridPredictor(model_path)
    
    with open(pgn_file) as f:
        game = chess.pgn.read_game(f)
    
    moves = []
    board = chess.Board()
    
    print(f"=== GAME STYLE ANALYSIS ===")
    print(f"Player Rating: {player_rating}")
    
    style_agreements = 0
    engine_agreements = 0
    total_moves = 0
    
    for move in game.mainline_moves():
        if len(moves) >= 10:  # Start analysis after opening
            result = predictor.predict_next_move(moves, player_rating, show_analysis=False)
            
            actual_move = str(move)
            
            # Check if actual move matches predictions
            style_match = any(actual_move == pred[0] for pred in result['hybrid_predictions'][:3])
            engine_match = actual_move in result['engine_suggestions'][:3]
            
            if style_match:
                style_agreements += 1
            if engine_match:
                engine_agreements += 1
            
            total_moves += 1
            
            if total_moves % 10 == 0:
                print(f"Move {total_moves}: Style={style_agreements/total_moves:.1%}, Engine={engine_agreements/total_moves:.1%}")
        
        moves.append(str(move))
        board.push(move)
    
    print(f"\n=== FINAL ANALYSIS ===")
    print(f"Style Agreement: {style_agreements/total_moves:.1%}")
    print(f"Engine Agreement: {engine_agreements/total_moves:.1%}")
    print(f"Playing Style: {'Engine-like' if engine_agreements/total_moves > 0.7 else 'Human-like'}")

if __name__ == "__main__":
    # Example usage
    model_path = "models/hybrid_chess_model.pt"
    
    # Analyze a position
    game_moves = ["e4", "e5", "Nf3", "Nc6", "Bb5"]
    player_rating = 1500
    
    predictor = HybridPredictor(model_path)
    result = predictor.predict_next_move(game_moves, player_rating)
