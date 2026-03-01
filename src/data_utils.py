import chess
import chess.pgn
import torch
from torch.utils.data import Dataset
import os
import re

class MoveTokenizer:
    def __init__(self):
        self.move_to_id = {}
        self.id_to_move = {}
        self.vocab_size = 0
        
    def build_vocab(self, pgn_files):
        moves = set()
        for pgn_file in pgn_files:
            with open(pgn_file, 'r') as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    for move in game.mainline_moves():
                        moves.add(str(move))
        
        # Add special tokens
        moves.update(['<PAD>', '<START>', '<END>'])
        
        for i, move in enumerate(sorted(moves)):
            self.move_to_id[move] = i
            self.id_to_move[i] = move
        
        self.vocab_size = len(moves)
    
    def build_default_vocab(self):
        """Build a default vocabulary with all legal chess moves"""
        moves = set(['<PAD>', '<START>', '<END>'])
        
        # Generate all possible chess moves in UCI format
        board = chess.Board()
        for square in chess.SQUARES:
            for target in chess.SQUARES:
                move = chess.Move(square, target)
                moves.add(str(move))
                # Add promotions
                for piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    promo_move = chess.Move(square, target, promotion=piece)
                    moves.add(str(promo_move))
        
        for i, move in enumerate(sorted(moves)):
            self.move_to_id[move] = i
            self.id_to_move[i] = move
        
        self.vocab_size = len(moves)
        
    def encode(self, moves):
        return [self.move_to_id.get(move, 0) for move in moves]
    
    def decode(self, ids):
        return [self.id_to_move.get(id, '<UNK>') for id in ids]

class ChessDataset(Dataset):
    def __init__(self, pgn_files, tokenizer, player_mapping, max_seq_len=50):
        self.data = []
        self.tokenizer = tokenizer
        self.player_mapping = player_mapping
        self.max_seq_len = max_seq_len
        
        for pgn_file in pgn_files:
            self._process_pgn(pgn_file)
    
    def _process_pgn(self, pgn_file):
        with open(pgn_file, 'r') as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                white_player = game.headers.get('White', 'Unknown')
                black_player = game.headers.get('Black', 'Unknown')
                
                moves = [str(move) for move in game.mainline_moves()]
                
                # Create training examples
                for i in range(1, len(moves)):
                    sequence = moves[:i]
                    target = moves[i]
                    
                    # Determine current player
                    current_player = white_player if i % 2 == 1 else black_player
                    
                    if current_player in self.player_mapping:
                        self.data.append({
                            'sequence': sequence,
                            'target': target,
                            'player': current_player
                        })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Encode sequence
        sequence = self.tokenizer.encode(item['sequence'])
        target = self.tokenizer.move_to_id.get(item['target'], 0)
        player_id = self.player_mapping[item['player']]
        
        # Pad sequence
        if len(sequence) > self.max_seq_len:
            sequence = sequence[-self.max_seq_len:]
        else:
            sequence = [0] * (self.max_seq_len - len(sequence)) + sequence
        
        return {
            'sequence': torch.tensor(sequence, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long),
            'player_id': torch.tensor(player_id, dtype=torch.long)
        }

def load_pgn_files(data_dir):
    pgn_files = []
    for file in os.listdir(data_dir):
        if file.endswith('.pgn'):
            pgn_files.append(os.path.join(data_dir, file))
    return pgn_files

def create_player_mapping(pgn_files):
    players = set()
    for pgn_file in pgn_files:
        with open(pgn_file, 'r') as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                players.add(game.headers.get('White', 'Unknown'))
                players.add(game.headers.get('Black', 'Unknown'))
    
    return {player: i for i, player in enumerate(sorted(players))}
