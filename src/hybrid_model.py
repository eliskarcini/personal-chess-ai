import torch
import torch.nn as nn
import chess
import chess.engine
from stockfish import Stockfish
import numpy as np

class ChessEngineIntegration:
    def __init__(self, engine_path="/usr/local/bin/stockfish", depth=10):
        self.stockfish = Stockfish(path=engine_path, depth=depth)
    
    def get_top_moves(self, fen, top_k=5):
        """Get top K moves with evaluations from engine"""
        self.stockfish.set_fen_position(fen)
        top_moves = self.stockfish.get_top_moves(top_k)
        
        moves_data = []
        for move_data in top_moves:
            moves_data.append({
                'move': move_data['Move'],
                'centipawn': move_data.get('Centipawn', 0),
                'mate': move_data.get('Mate', None)
            })
        return moves_data

class HybridChessModel(nn.Module):
    def __init__(self, vocab_size=4096, d_model=256, nhead=8, num_layers=4, 
                 rating_bins=10, engine_features=5):
        super().__init__()
        self.d_model = d_model
        
        # Move sequence encoding
        self.move_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_positional_encoding(d_model)
        
        # Player characteristics
        self.rating_embedding = nn.Embedding(rating_bins, 64)
        
        # Engine features (top moves, evaluations)
        self.engine_projection = nn.Linear(engine_features, 64)
        
        # Transformer for sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            dropout=0.2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Style vs Engine balance network
        self.style_head = nn.Linear(d_model + 64, vocab_size)  # Player style prediction
        self.engine_head = nn.Linear(64, vocab_size)           # Engine move ranking
        self.balance_net = nn.Linear(d_model + 64 + 64, 1)     # Mixing weight
        
        self.dropout = nn.Dropout(0.2)
        
    def _create_positional_encoding(self, d_model, max_len=200):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, move_sequence, rating_bin, engine_features, attention_mask=None):
        # Encode move sequence
        x = self.move_embedding(move_sequence) * np.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :x.size(1)].to(x.device)
        x = self.dropout(x)
        
        # Transform sequence
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        last_move = x[:, -1, :]
        
        # Player characteristics
        rating_emb = self.rating_embedding(rating_bin)
        engine_emb = self.engine_projection(engine_features)
        
        # Style prediction (what player would likely do)
        style_features = torch.cat([last_move, rating_emb], dim=-1)
        style_logits = self.style_head(style_features)
        
        # Engine move ranking
        engine_logits = self.engine_head(engine_emb)
        
        # Balance between style and engine
        balance_features = torch.cat([last_move, rating_emb, engine_emb], dim=-1)
        alpha = torch.sigmoid(self.balance_net(balance_features))
        
        # Hybrid prediction
        hybrid_logits = alpha * style_logits + (1 - alpha) * engine_logits
        
        return {
            'hybrid_logits': hybrid_logits,
            'style_logits': style_logits,
            'engine_logits': engine_logits,
            'balance_weight': alpha
        }

class HybridTrainer:
    def __init__(self, model, engine_integration, device='cuda'):
        self.model = model
        self.engine = engine_integration
        self.device = device
        
    def prepare_training_data(self, game_data):
        """Prepare data with engine evaluations"""
        enhanced_data = []
        
        for game in game_data:
            board = chess.Board()
            moves = game['moves']
            rating = game['rating']
            
            for i, move in enumerate(moves[:-1]):
                board.push_san(move)
                
                # Get engine top moves for current position
                engine_moves = self.engine.get_top_moves(board.fen())
                
                # Create engine feature vector
                engine_features = self._create_engine_features(engine_moves, moves[i+1])
                
                enhanced_data.append({
                    'sequence': moves[:i+1],
                    'target': moves[i+1],
                    'rating_bin': self._rating_to_bin(rating),
                    'engine_features': engine_features
                })
                
        return enhanced_data
    
    def _create_engine_features(self, engine_moves, actual_move):
        """Create feature vector from engine analysis"""
        features = np.zeros(5)
        
        if engine_moves:
            # Best move evaluation
            features[0] = engine_moves[0]['centipawn'] / 100.0  # Normalize
            
            # Find actual move in engine suggestions
            for i, move_data in enumerate(engine_moves):
                if move_data['move'] == actual_move:
                    features[1] = i / len(engine_moves)  # Rank of actual move
                    features[2] = move_data['centipawn'] / 100.0
                    break
            else:
                features[1] = 1.0  # Not in top moves
                features[2] = -2.0  # Penalty
            
            # Evaluation spread
            if len(engine_moves) > 1:
                evals = [m['centipawn'] for m in engine_moves]
                features[3] = (max(evals) - min(evals)) / 100.0
            
            # Number of good alternatives
            features[4] = len([m for m in engine_moves if m['centipawn'] > -50]) / 5.0
        
        return features
    
    def _rating_to_bin(self, rating):
        """Convert rating to bin (0-9 for different skill levels)"""
        return min(9, max(0, (rating - 800) // 200))

def create_hybrid_model(vocab_size=4096, rating_bins=10):
    return HybridChessModel(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_layers=4,
        rating_bins=rating_bins,
        engine_features=5
    )
