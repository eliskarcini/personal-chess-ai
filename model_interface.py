import torch
import chess
import chess.pgn
import pickle
import os
from src.hybrid_model import HybridChessModel
from src.model import create_model
from src.data_utils import MoveTokenizer
from stockfish_analyzer import StockfishAnalyzer

class ChessModelInterface:
    def __init__(self, basic_model_path, hybrid_model_path, pgn_data_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading models on device: {self.device}")
        
        # Initialize Stockfish analyzer
        self.stockfish = StockfishAnalyzer()
        
        # Create tokenizer from training data
        self.tokenizer = self._create_tokenizer(pgn_data_path)
        
        # Load basic model
        self.basic_model = self._load_basic_model(basic_model_path)
        
        # Load hybrid model  
        self.hybrid_model = self._load_hybrid_model(hybrid_model_path)
        
        print("Models loaded successfully!")
    
    def _create_tokenizer(self, pgn_data_path):
        """Create tokenizer from saved vocab or PGN files"""
        print("Loading tokenizer...")
        tokenizer = MoveTokenizer()
        
        # Try to load saved vocab first
        vocab_path = 'models/tokenizer_vocab.pkl'
        if os.path.exists(vocab_path):
            import pickle
            with open(vocab_path, 'rb') as f:
                vocab_data = pickle.load(f)
                tokenizer.move_to_id = vocab_data['move_to_id']
                tokenizer.id_to_move = vocab_data['id_to_move']
                tokenizer.vocab_size = vocab_data['vocab_size']
            print(f"✅ Loaded saved vocab with {tokenizer.vocab_size} moves")
        elif pgn_data_path and os.path.exists(pgn_data_path):
            # Fallback to building from PGN files
            pgn_files = [os.path.join(pgn_data_path, f) for f in os.listdir(pgn_data_path) if f.endswith('.pgn')]
            if pgn_files:
                tokenizer.build_vocab(pgn_files)
                print(f"Built vocab from PGN files: {tokenizer.vocab_size} moves")
        else:
            # Last resort: build default vocab
            tokenizer.build_default_vocab()
            print(f"⚠️  Using default vocab: {tokenizer.vocab_size} moves")
        
        return tokenizer
    
    def _load_basic_model(self, model_path):
        """Load basic chess model"""
        print("Loading basic model...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model = create_model(
            num_players=checkpoint['num_players'], 
            vocab_size=checkpoint['vocab_size']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model
    
    def _load_hybrid_model(self, model_path):
        """Load hybrid chess model"""
        print("Loading hybrid model...")
        model = HybridChessModel(vocab_size=4096, rating_bins=10)
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
        model.to(self.device)
        model.eval()
        return model
    
    def predict_move(self, board, move_history, player_rating=1500, use_hybrid=True):
        """
        Predict next move given current board position
        
        Args:
            board: chess.Board object
            move_history: list of moves in string format (e.g., ['e4', 'e5', 'Nf3'])
            player_rating: player's rating for hybrid model
            use_hybrid: whether to use hybrid model or basic model
        
        Returns:
            dict with predictions and analysis
        """
        # Convert move history to tokens
        if len(move_history) == 0:
            return self._get_opening_move()
        
        # Encode moves
        sequence_tokens = self.tokenizer.encode(move_history)
        
        # Pad to model input size
        max_len = 50 if not use_hybrid else 20
        if len(sequence_tokens) > max_len:
            sequence_tokens = sequence_tokens[-max_len:]
        else:
            sequence_tokens = [0] * (max_len - len(sequence_tokens)) + sequence_tokens
        
        sequence_tensor = torch.tensor([sequence_tokens], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            if use_hybrid:
                return self._predict_hybrid(sequence_tensor, board, player_rating)
            else:
                return self._predict_basic(sequence_tensor, board)
    
    def _predict_basic(self, sequence_tensor, board):
        """Get prediction from basic model"""
        output = self.basic_model(sequence_tensor, torch.tensor([0]).to(self.device))
        probabilities = torch.softmax(output, dim=-1)[0]
        
        # Get legal moves and their probabilities
        legal_moves = list(board.legal_moves)
        move_scores = []
        
        for move in legal_moves:
            move_str = str(move)
            if move_str in self.tokenizer.move_to_id:
                token_id = self.tokenizer.move_to_id[move_str]
                score = probabilities[token_id].item()
            else:
                score = 0.0
            move_scores.append((move_str, score))
        
        # Sort by score
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'model': 'basic',
            'top_moves': move_scores[:5],
            'best_move': move_scores[0][0] if move_scores else None
        }
    
    def _predict_hybrid(self, sequence_tensor, board, player_rating):
        """Get prediction from hybrid model"""
        rating_bin = min(9, max(0, (player_rating - 800) // 200))
        rating_tensor = torch.tensor([rating_bin]).to(self.device)
        
        # Get real engine features instead of dummy zeros
        engine_features_array = self.stockfish.analyze_position(board)
        engine_features = torch.tensor(engine_features_array.reshape(1, -1), dtype=torch.float32).to(self.device)
        
        output = self.hybrid_model(sequence_tensor, rating_tensor, engine_features)
        
        style_probs = torch.softmax(output['style_logits'], dim=-1)[0]
        engine_probs = torch.softmax(output['engine_logits'], dim=-1)[0]
        hybrid_probs = torch.softmax(output['hybrid_logits'], dim=-1)[0]
        balance_weight = output['balance_weight'].item()
        
        # Get legal moves and their probabilities
        legal_moves = list(board.legal_moves)
        move_scores = []
        
        for move in legal_moves:
            move_str = str(move)
            if move_str in self.tokenizer.move_to_id:
                token_id = self.tokenizer.move_to_id[move_str]
                style_score = style_probs[token_id].item()
                engine_score = engine_probs[token_id].item()
                hybrid_score = hybrid_probs[token_id].item()
            else:
                style_score = engine_score = hybrid_score = 0.0
            
            move_scores.append((move_str, hybrid_score, style_score, engine_score))
        
        # Sort by hybrid score
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get Stockfish best moves for comparison
        stockfish_moves = self.stockfish.get_best_moves(board, 3)
        
        return {
            'model': 'hybrid',
            'top_moves': move_scores[:5],
            'best_move': move_scores[0][0] if move_scores else None,
            'balance_weight': balance_weight,
            'rating_bin': rating_bin,
            'engine_features': engine_features_array.tolist(),
            'stockfish_moves': stockfish_moves
        }
    
    def _get_opening_move(self):
        """Return common opening moves for empty board"""
        opening_moves = ['e4', 'd4', 'Nf3', 'c4']
        return {
            'model': 'opening',
            'top_moves': [(move, 0.25) for move in opening_moves],
            'best_move': 'e4'
        }
    
    def analyze_position(self, board, move_history, player_rating=1500):
        """Compare basic vs hybrid model predictions"""
        basic_result = self.predict_move(board, move_history, player_rating, use_hybrid=False)
        hybrid_result = self.predict_move(board, move_history, player_rating, use_hybrid=True)
        
        return {
            'basic': basic_result,
            'hybrid': hybrid_result,
            'position_fen': board.fen()
        }

# Test the interface
if __name__ == "__main__":
    # Initialize the interface
    interface = ChessModelInterface(
        basic_model_path="models/best_model.pth",
        hybrid_model_path="models/hybrid_chess_model.pt", 
        pgn_data_path="data/raw_games"
    )
    
    # Test with a position after a few moves
    board = chess.Board()
    move_history = ['e4', 'e5', 'Nf3']
    
    # Apply moves to board
    for move_str in move_history:
        move = board.parse_san(move_str)
        board.push(move)
    
    print("\n=== TESTING MODEL INTERFACE ===")
    print(f"Position after: {' '.join(move_history)}")
    print(f"FEN: {board.fen()}")
    
    result = interface.analyze_position(board, move_history, player_rating=1500)
    
    print(f"\nBasic model top moves:")
    for move, score in result['basic']['top_moves']:
        print(f"  {move}: {score:.4f}")
    
    print(f"\nHybrid model top moves:")
    for move, hybrid_score, style_score, engine_score in result['hybrid']['top_moves']:
        print(f"  {move}: hybrid={hybrid_score:.4f}, style={style_score:.4f}, engine={engine_score:.4f}")
    
    if 'balance_weight' in result['hybrid']:
        print(f"\nHybrid balance weight: {result['hybrid']['balance_weight']:.3f}")
        print(f"Rating bin: {result['hybrid']['rating_bin']}")
        print(f"Engine features: {result['hybrid']['engine_features']}")
        print(f"Stockfish best moves: {result['hybrid']['stockfish_moves']}")
    
    # Clean up
    interface.stockfish.close()
