import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ChessTransformer(nn.Module):
    def __init__(self, vocab_size=4096, d_model=256, nhead=8, num_layers=4, num_players=4):
        super().__init__()
        self.d_model = d_model
        self.move_embedding = nn.Embedding(vocab_size, d_model)
        self.player_embedding = nn.Embedding(num_players, 64)
        self.pos_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=512,
            dropout=0.2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output = nn.Linear(d_model + 64, vocab_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, move_sequence, player_id, attention_mask=None):
        x = self.move_embedding(move_sequence) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        
        last_move = x[:, -1, :]
        player_emb = self.player_embedding(player_id)
        
        combined = torch.cat([last_move, player_emb], dim=-1)
        return self.output(combined)

def create_model(num_players=4, vocab_size=4096):
    return ChessTransformer(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_layers=4,
        num_players=num_players
    )
