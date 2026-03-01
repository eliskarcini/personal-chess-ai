import torch
import chess
import numpy as np
from model_interface import ChessModelInterface
from stockfish_analyzer import StockfishAnalyzer

class EnhancedChessAI:
    def __init__(self, basic_model_path, hybrid_model_path, pgn_data_path):
        """Enhanced AI that combines strong engine play with style preferences"""
        self.model_interface = ChessModelInterface(
            basic_model_path, hybrid_model_path, pgn_data_path
        )
        self.stockfish = StockfishAnalyzer()
        
    def get_enhanced_move(self, board, move_history, player_rating=1500, strength_level=1500):
        """
        Get move using enhanced engine integration
        
        Args:
            board: Current chess position
            move_history: List of moves played
            player_rating: Player's rating for style
            strength_level: AI strength (800-2800)
        
        Returns:
            Best move with analysis
        """
        # Get engine analysis (the chess brain)
        engine_moves = self.stockfish.get_best_moves(board, num_moves=5)
        
        if not engine_moves:
            # Fallback to model if engine fails
            result = self.model_interface.analyze_position(board, move_history, player_rating)
            return result['hybrid']
        
        # Get model preferences (the style brain)
        model_result = self.model_interface.analyze_position(board, move_history, player_rating)
        model_moves = model_result['hybrid']['top_moves']
        
        # Enhanced move selection based on strength level
        if strength_level >= 2000:
            # Strong player: 95% engine, 5% style
            return self._select_move(engine_moves, model_moves, engine_weight=0.95)
        elif strength_level >= 1500:
            # Intermediate player: 80% engine, 20% style  
            return self._select_move(engine_moves, model_moves, engine_weight=0.80)
        elif strength_level >= 1200:
            # Beginner player: 60% engine, 40% style
            return self._select_move(engine_moves, model_moves, engine_weight=0.60)
        else:
            # Very weak: 40% engine, 60% style (more human-like mistakes)
            return self._select_move(engine_moves, model_moves, engine_weight=0.40)
    
    def _select_move(self, engine_moves, model_moves, engine_weight=0.8):
        """Select move combining engine strength with model style"""
        
        # Create move scoring system
        move_scores = {}
        
        # Score engine moves (tactical strength)
        for i, (move, centipawns) in enumerate(engine_moves):
            # Higher score for better engine evaluation
            engine_score = max(0, 100 - i * 20)  # 100, 80, 60, 40, 20
            
            # Bonus for clearly winning moves
            if centipawns > 200:  # Winning material
                engine_score += 50
            elif centipawns > 50:   # Small advantage
                engine_score += 20
            
            move_scores[move] = move_scores.get(move, 0) + engine_score * engine_weight
        
        # Score model moves (style preference)
        for move_data in model_moves[:5]:
            move = move_data[0]
            model_score = move_data[1] * 1000  # Scale up probability
            
            move_scores[move] = move_scores.get(move, 0) + model_score * (1 - engine_weight)
        
        # Find best combined move
        if not move_scores:
            return {'best_move': engine_moves[0][0] if engine_moves else None}
        
        best_move = max(move_scores.items(), key=lambda x: x[1])
        
        # Create analysis
        engine_best = engine_moves[0][0] if engine_moves else None
        model_best = model_moves[0][0] if model_moves else None
        
        analysis = {
            'best_move': best_move[0],
            'combined_score': best_move[1],
            'engine_best': engine_best,
            'model_best': model_best,
            'engine_weight': engine_weight,
            'move_scores': dict(list(move_scores.items())[:5])  # Top 5 for display
        }
        
        return analysis
    
    def analyze_position_enhanced(self, board, move_history, player_rating=1500, strength_level=1500):
        """Get enhanced position analysis"""
        
        # Get engine evaluation
        engine_features = self.stockfish.analyze_position(board)
        engine_moves = self.stockfish.get_best_moves(board, 3)
        
        # Get model analysis
        model_result = self.model_interface.analyze_position(board, move_history, player_rating)
        
        # Get enhanced move recommendation
        enhanced_move = self.get_enhanced_move(board, move_history, player_rating, strength_level)
        
        return {
            'enhanced_move': enhanced_move,
            'engine_eval': engine_features[0] if len(engine_features) > 0 else 0,
            'engine_moves': engine_moves,
            'model_analysis': model_result,
            'position_fen': board.fen(),
            'strength_level': strength_level
        }

# Test the enhanced AI
if __name__ == "__main__":
    print("=== TESTING ENHANCED CHESS AI ===")
    
    ai = EnhancedChessAI(
        basic_model_path="models/best_model.pth",
        hybrid_model_path="models/hybrid_chess_model.pt",
        pgn_data_path="data/raw_games"
    )
    
    # Test position with tactical opportunity
    board = chess.Board("rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3")
    move_history = ["e4", "e5", "Bc4", "Nf6"]
    
    print(f"Test position: {board.fen()}")
    print("Move history:", " ".join(move_history))
    
    # Test different strength levels
    for strength in [1200, 1500, 2000]:
        print(f"\n--- Strength Level: {strength} ---")
        result = ai.analyze_position_enhanced(board, move_history, 1500, strength)
        
        print(f"Enhanced move: {result['enhanced_move']['best_move']}")
        print(f"Engine best: {result['enhanced_move']['engine_best']}")
        print(f"Model best: {result['enhanced_move']['model_best']}")
        print(f"Engine weight: {result['enhanced_move']['engine_weight']:.1%}")
    
    ai.stockfish.close()
