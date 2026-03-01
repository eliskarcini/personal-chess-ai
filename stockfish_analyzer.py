import chess
import chess.engine
from stockfish import Stockfish
import numpy as np

class StockfishAnalyzer:
    def __init__(self, stockfish_path="/opt/homebrew/bin/stockfish", depth=10):
        """Initialize Stockfish analyzer"""
        try:
            # Try python-chess engine first (more reliable)
            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            self.use_python_chess = True
            print("Using python-chess Stockfish integration")
        except:
            try:
                # Fallback to stockfish library
                self.stockfish = Stockfish(path=stockfish_path, depth=depth)
                self.use_python_chess = False
                print("Using stockfish library integration")
            except Exception as e:
                print(f"Warning: Could not initialize Stockfish: {e}")
                self.engine = None
                self.stockfish = None
                self.use_python_chess = None
        
        self.depth = depth
    
    def analyze_position(self, board, time_limit=0.1):
        """
        Analyze chess position and return engine features
        
        Returns:
            numpy array of 5 features:
            [position_eval, best_move_rank, move_quality, eval_spread, num_good_moves]
        """
        if self.use_python_chess and self.engine:
            return self._analyze_with_python_chess(board, time_limit)
        elif not self.use_python_chess and self.stockfish:
            return self._analyze_with_stockfish_lib(board)
        else:
            # Return dummy features if no engine available
            return np.zeros(5)
    
    def _analyze_with_python_chess(self, board, time_limit):
        """Analyze using python-chess engine"""
        try:
            # Get top moves with analysis
            info = self.engine.analyse(board, chess.engine.Limit(time=time_limit), multipv=5)
            
            if not info:
                return np.zeros(5)
            
            features = np.zeros(5)
            
            # Feature 0: Position evaluation (centipawns, normalized)
            if 'score' in info[0]:
                score = info[0]['score'].relative
                if score.is_mate():
                    features[0] = 10.0 if score.mate() > 0 else -10.0
                else:
                    features[0] = score.score() / 100.0  # Convert to pawns
            
            # Feature 1: Number of analyzed moves
            features[1] = len(info) / 5.0  # Normalize to 0-1
            
            # Feature 2: Best move confidence (depth reached)
            if 'depth' in info[0]:
                features[2] = min(info[0]['depth'] / 20.0, 1.0)  # Normalize depth
            
            # Feature 3: Evaluation spread (difference between best and worst)
            if len(info) > 1:
                scores = []
                for analysis in info:
                    if 'score' in analysis:
                        score = analysis['score'].relative
                        if not score.is_mate():
                            scores.append(score.score())
                
                if len(scores) > 1:
                    features[3] = (max(scores) - min(scores)) / 100.0
            
            # Feature 4: Number of good moves (within 50 centipawns of best)
            good_moves = 0
            if len(info) > 0 and 'score' in info[0]:
                best_score = info[0]['score'].relative
                if not best_score.is_mate():
                    best_cp = best_score.score()
                    for analysis in info[1:]:
                        if 'score' in analysis:
                            score = analysis['score'].relative
                            if not score.is_mate() and abs(score.score() - best_cp) <= 50:
                                good_moves += 1
            
            features[4] = good_moves / 4.0  # Normalize (max 4 additional good moves)
            
            return features
            
        except Exception as e:
            print(f"Engine analysis error: {e}")
            return np.zeros(5)
    
    def _analyze_with_stockfish_lib(self, board):
        """Analyze using stockfish library"""
        try:
            self.stockfish.set_fen_position(board.fen())
            
            # Get top moves
            top_moves = self.stockfish.get_top_moves(5)
            if not top_moves:
                return np.zeros(5)
            
            features = np.zeros(5)
            
            # Feature 0: Position evaluation
            evaluation = self.stockfish.get_evaluation()
            if evaluation['type'] == 'cp':
                features[0] = evaluation['value'] / 100.0
            elif evaluation['type'] == 'mate':
                features[0] = 10.0 if evaluation['value'] > 0 else -10.0
            
            # Feature 1: Number of good moves
            features[1] = len(top_moves) / 5.0
            
            # Feature 2: Best move confidence (always 1.0 for stockfish lib)
            features[2] = 1.0
            
            # Feature 3: Evaluation spread
            if len(top_moves) > 1:
                evals = [move.get('Centipawn', 0) for move in top_moves]
                features[3] = (max(evals) - min(evals)) / 100.0
            
            # Feature 4: Number of moves within 50cp of best
            if len(top_moves) > 0:
                best_eval = top_moves[0].get('Centipawn', 0)
                good_moves = sum(1 for move in top_moves[1:] 
                               if abs(move.get('Centipawn', 0) - best_eval) <= 50)
                features[4] = good_moves / 4.0
            
            return features
            
        except Exception as e:
            print(f"Stockfish library error: {e}")
            return np.zeros(5)
    
    def get_best_moves(self, board, num_moves=5):
        """Get best moves from engine"""
        if self.use_python_chess and self.engine:
            try:
                info = self.engine.analyse(board, chess.engine.Limit(time=0.1), multipv=num_moves)
                moves = []
                for analysis in info:
                    if 'pv' in analysis and len(analysis['pv']) > 0:
                        move = analysis['pv'][0]
                        score = analysis.get('score', chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE))
                        moves.append((str(move), score.relative.score() if not score.relative.is_mate() else 0))
                return moves
            except:
                return []
        elif not self.use_python_chess and self.stockfish:
            try:
                self.stockfish.set_fen_position(board.fen())
                top_moves = self.stockfish.get_top_moves(num_moves)
                return [(move['Move'], move.get('Centipawn', 0)) for move in top_moves]
            except:
                return []
        else:
            return []
    
    def close(self):
        """Clean up engine resources"""
        if self.use_python_chess and self.engine:
            self.engine.quit()

# Test the analyzer
if __name__ == "__main__":
    analyzer = StockfishAnalyzer()
    
    # Test position
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2")
    
    print("=== STOCKFISH ANALYSIS TEST ===")
    print(f"Position: {board.fen()}")
    
    features = analyzer.analyze_position(board)
    print(f"Engine features: {features}")
    
    best_moves = analyzer.get_best_moves(board, 3)
    print(f"Best moves: {best_moves}")
    
    analyzer.close()
