import chess
import chess.pgn
from enhanced_ai import EnhancedChessAI
from stockfish_analyzer import StockfishAnalyzer
import time

class ChessEngine:
    def __init__(self, basic_model_path, hybrid_model_path, pgn_data_path):
        """Initialize the chess engine with enhanced AI models"""
        self.enhanced_ai = EnhancedChessAI(
            basic_model_path, hybrid_model_path, pgn_data_path
        )
        self.board = chess.Board()
        self.move_history = []
        self.game_pgn = []
        
    def reset_game(self):
        """Start a new game"""
        self.board = chess.Board()
        self.move_history = []
        self.game_pgn = []
        print("New game started!")
        print(f"Position: {self.board}")
    
    def make_move(self, move_str):
        """
        Make a move on the board
        
        Args:
            move_str: Move in algebraic notation (e.g., 'e4', 'Nf3', 'O-O')
        
        Returns:
            dict with move result and game state
        """
        try:
            # Parse and validate move
            move = self.board.parse_san(move_str)
            
            if move not in self.board.legal_moves:
                return {
                    'success': False,
                    'error': f"Illegal move: {move_str}",
                    'legal_moves': [str(m) for m in self.board.legal_moves]
                }
            
            # Make the move
            self.board.push(move)
            self.move_history.append(move_str)
            self.game_pgn.append(move_str)
            
            # Check game state
            game_state = self._get_game_state()
            
            return {
                'success': True,
                'move': move_str,
                'board': str(self.board),
                'fen': self.board.fen(),
                'game_state': game_state,
                'move_number': len(self.move_history)
            }
            
        except ValueError as e:
            return {
                'success': False,
                'error': f"Invalid move format: {move_str}. {str(e)}",
                'legal_moves': [str(m) for m in self.board.legal_moves]
            }
    
    def get_ai_move(self, player_rating=1500, use_hybrid=True, thinking_time=1.0, strength_level=1500):
        """
        Get AI's best move using enhanced engine integration
        """
        if self.board.is_game_over():
            return {
                'success': False,
                'error': 'Game is over',
                'game_state': self._get_game_state()
            }
        
        start_time = time.time()
        
        # Use enhanced AI instead of basic model interface
        try:
            analysis = self.enhanced_ai.get_enhanced_move(
                self.board, self.move_history, player_rating, strength_level
            )
            
            best_move = analysis['best_move']
            
            if best_move:
                # Validate and make the move
                try:
                    move = self.board.parse_san(best_move)
                    if move in self.board.legal_moves:
                        move_result = self.make_move(best_move)
                        
                        return {
                            'success': True,
                            'ai_move': best_move,
                            'analysis': analysis,
                            'thinking_time': time.time() - start_time,
                            'move_result': move_result
                        }
                except:
                    pass
            
            # Fallback to first legal move
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                fallback_move = str(legal_moves[0])
                move_result = self.make_move(fallback_move)
                
                return {
                    'success': True,
                    'ai_move': fallback_move,
                    'analysis': {'note': 'Fallback move'},
                    'thinking_time': time.time() - start_time,
                    'move_result': move_result
                }
                
        except Exception as e:
            print(f"Enhanced AI error: {e}")
            # Ultimate fallback
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                fallback_move = str(legal_moves[0])
                move_result = self.make_move(fallback_move)
                
                return {
                    'success': True,
                    'ai_move': fallback_move,
                    'analysis': {'note': f'Error fallback: {str(e)}'},
                    'thinking_time': time.time() - start_time,
                    'move_result': move_result
                }
        
        return {
            'success': False,
            'error': 'No legal moves available'
        }
    
    def _would_create_repetition(self, proposed_move, last_move, second_last_move):
        """Check if proposed move would create immediate repetition"""
        if not last_move or not second_last_move or len(self.move_history) < 2:
            return False
        
        try:
            # Create a copy of the board to avoid modifying the original
            temp_board = chess.Board(self.board.fen())
            
            # Only check if we have enough move history
            if len(self.move_history) < 1:
                return False
            
            # Get the actual move objects
            proposed_move_obj = temp_board.parse_san(proposed_move)
            proposed_from = chess.square_name(proposed_move_obj.from_square)
            proposed_to = chess.square_name(proposed_move_obj.to_square)
            
            # Check if we can safely go back one move
            if temp_board.move_stack:
                temp_board.pop()  # Go back one move
                last_move_obj = temp_board.parse_san(last_move)
                last_from = chess.square_name(last_move_obj.from_square)
                last_to = chess.square_name(last_move_obj.to_square)
                
                # Check if it's the same piece moving back and forth
                if (last_to == proposed_from and last_from == proposed_to):
                    print(f"Avoiding repetition: {proposed_move} would reverse {last_move}")
                    return True
            
            # Also check for exact move repetition
            if proposed_move == last_move:
                print(f"Avoiding exact move repetition: {proposed_move}")
                return True
                
        except Exception as e:
            print(f"Repetition check error: {e}")
            # Don't block moves if there's an error
            pass
        
        return False
    
    def _get_game_state(self):
        """Get current game state"""
        if self.board.is_checkmate():
            winner = "White" if self.board.turn == chess.BLACK else "Black"
            return {'status': 'checkmate', 'winner': winner}
        elif self.board.is_stalemate():
            return {'status': 'stalemate', 'result': 'draw'}
        elif self.board.is_insufficient_material():
            return {'status': 'insufficient_material', 'result': 'draw'}
        elif self.board.is_seventyfive_moves():
            return {'status': '75_move_rule', 'result': 'draw'}
        elif self.board.is_fivefold_repetition():
            return {'status': 'repetition', 'result': 'draw'}
        elif self.board.is_check():
            return {'status': 'check', 'turn': 'White' if self.board.turn else 'Black'}
        else:
            return {'status': 'playing', 'turn': 'White' if self.board.turn else 'Black'}
    
    def get_legal_moves(self):
        """Get all legal moves in current position"""
        return [str(move) for move in self.board.legal_moves]
    
    def get_position_analysis(self, player_rating=1500):
        """Get detailed position analysis"""
        result = self.model_interface.analyze_position(
            self.board, self.move_history, player_rating
        )
        
        return {
            'fen': self.board.fen(),
            'turn': 'White' if self.board.turn else 'Black',
            'legal_moves': self.get_legal_moves(),
            'move_count': len(self.move_history),
            'basic_analysis': result['basic'],
            'hybrid_analysis': result['hybrid'],
            'game_state': self._get_game_state()
        }
    
    def save_game(self, filename, white_player="Human", black_player="AI"):
        """Save game as PGN file"""
        game = chess.pgn.Game()
        game.headers["White"] = white_player
        game.headers["Black"] = black_player
        game.headers["Result"] = self._get_pgn_result()
        
        node = game
        board = chess.Board()
        
        for move_str in self.game_pgn:
            move = board.parse_san(move_str)
            node = node.add_variation(move)
            board.push(move)
        
        with open(filename, 'w') as f:
            print(game, file=f)
        
        print(f"Game saved to {filename}")
    
    def _get_pgn_result(self):
        """Get PGN result string"""
        state = self._get_game_state()
        if state['status'] == 'checkmate':
            return "1-0" if state['winner'] == "White" else "0-1"
        elif state['status'] in ['stalemate', 'insufficient_material', '75_move_rule', 'repetition']:
            return "1/2-1/2"
        else:
            return "*"

# Test the chess engine
if __name__ == "__main__":
    print("=== CHESS ENGINE TEST ===")
    
    engine = ChessEngine(
        basic_model_path="models/best_model.pth",
        hybrid_model_path="models/hybrid_chess_model.pt",
        pgn_data_path="data/raw_games"
    )
    
    # Test a few moves
    print("\n1. Testing human move:")
    result = engine.make_move("e4")
    print(f"Move result: {result['success']}")
    print(f"Board:\n{result['board']}")
    
    print("\n2. Testing AI response:")
    ai_result = engine.get_ai_move(player_rating=1500, use_hybrid=True)
    print(f"AI played: {ai_result['ai_move']}")
    print(f"Thinking time: {ai_result['thinking_time']:.2f}s")
    
    print("\n3. Position analysis:")
    analysis = engine.get_position_analysis()
    print(f"Turn: {analysis['turn']}")
    print(f"Legal moves: {analysis['legal_moves'][:5]}...")  # Show first 5
    
    # Clean up
    engine.model_interface.stockfish.close()
