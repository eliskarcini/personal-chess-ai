import torch
import torch.nn as nn
import math
import numpy as np

class MaiaStyleModel(nn.Module):
    """
    Maia4All-inspired model with individual player embeddings
    Combines population-level and individual-level modeling
    """
    
    def __init__(self, vocab_size=4096, d_model=256, nhead=8, num_layers=4, 
                 num_population_bins=11, max_individual_players=10000, 
                 engine_features=5):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_individual_players = max_individual_players
        
        # Move sequence encoding
        self.move_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_positional_encoding(d_model)
        
        # Population embeddings (rating-based)
        self.population_embeddings = nn.Embedding(num_population_bins, 64)
        
        # Individual player embeddings (for personalization)
        self.individual_embeddings = nn.Embedding(max_individual_players, 64)
        
        # Engine features integration
        self.engine_projection = nn.Linear(engine_features, 64)
        
        # Transformer backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            dropout=0.2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Multi-head prediction system
        combined_dim = d_model + 64 + 64 + 64  # sequence + population + individual + engine
        
        self.style_head = nn.Linear(combined_dim, vocab_size)      # Personal style
        self.population_head = nn.Linear(combined_dim, vocab_size) # Population patterns
        self.engine_head = nn.Linear(combined_dim, vocab_size)     # Engine-guided moves
        
        # Adaptive mixing network
        self.mixing_network = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # 3 weights for style, population, engine
            nn.Softmax(dim=-1)
        )
        
        self.dropout = nn.Dropout(0.2)
        
    def _create_positional_encoding(self, d_model, max_len=200):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, move_sequence, population_id, individual_id=None, 
                engine_features=None, attention_mask=None):
        """
        Forward pass with multi-level embeddings
        
        Args:
            move_sequence: [batch_size, seq_len] - tokenized moves
            population_id: [batch_size] - rating bin (0-10)
            individual_id: [batch_size] - player ID (0-max_players, optional)
            engine_features: [batch_size, engine_features] - Stockfish analysis
            attention_mask: [batch_size, seq_len] - padding mask
        """
        batch_size, seq_len = move_sequence.shape
        
        # Encode move sequence
        x = self.move_embedding(move_sequence) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :seq_len].to(x.device)
        x = self.dropout(x)
        
        # Transform sequence
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        sequence_repr = x[:, -1, :]  # Last position representation
        
        # Population embedding (rating-based)
        pop_emb = self.population_embeddings(population_id)
        
        # Individual embedding (player-specific)
        if individual_id is not None:
            ind_emb = self.individual_embeddings(individual_id)
        else:
            # Use population embedding as fallback
            ind_emb = pop_emb.clone()
        
        # Engine features
        if engine_features is not None:
            eng_emb = self.engine_projection(engine_features)
        else:
            eng_emb = torch.zeros(batch_size, 64).to(x.device)
        
        # Combine all representations
        combined = torch.cat([sequence_repr, pop_emb, ind_emb, eng_emb], dim=-1)
        
        # Multi-head predictions
        style_logits = self.style_head(combined)
        population_logits = self.population_head(combined)
        engine_logits = self.engine_head(combined)
        
        # Adaptive mixing
        mixing_weights = self.mixing_network(combined)
        
        # Weighted combination
        hybrid_logits = (mixing_weights[:, 0:1] * style_logits + 
                        mixing_weights[:, 1:2] * population_logits + 
                        mixing_weights[:, 2:3] * engine_logits)
        
        return {
            'hybrid_logits': hybrid_logits,
            'style_logits': style_logits,
            'population_logits': population_logits,
            'engine_logits': engine_logits,
            'mixing_weights': mixing_weights,
            'individual_embedding': ind_emb,
            'population_embedding': pop_emb
        }
    
    def get_individual_embedding(self, individual_id):
        """Get individual player embedding for analysis"""
        return self.individual_embeddings(individual_id)
    
    def set_individual_embedding(self, individual_id, embedding):
        """Set individual player embedding (for initialization)"""
        with torch.no_grad():
            self.individual_embeddings.weight[individual_id] = embedding
    
    def freeze_backbone(self):
        """Freeze transformer backbone for individual fine-tuning"""
        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.move_embedding.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze transformer backbone"""
        for param in self.transformer.parameters():
            param.requires_grad = True
        for param in self.move_embedding.parameters():
            param.requires_grad = True

class PrototypeMatchingNetwork(nn.Module):
    """
    Network to match new players to existing prototypes
    Used for intelligent initialization of individual embeddings
    """
    
    def __init__(self, d_model=256, num_prototypes=200):
        super().__init__()
        self.d_model = d_model
        self.num_prototypes = num_prototypes
        
        # Position encoder (reuse from main model)
        self.position_encoder = nn.Sequential(
            nn.Linear(773, d_model),  # Chess position features
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Move sequence encoder
        self.move_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True),
            num_layers=2
        )
        
        # Style classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_prototypes)
        )
    
    def forward(self, move_sequences, position_features=None):
        """
        Match player style to prototypes
        
        Args:
            move_sequences: [batch_size, seq_len, move_features]
            position_features: [batch_size, seq_len, pos_features] (optional)
        """
        # Encode move sequences
        x = self.move_encoder(move_sequences)
        
        # Global average pooling
        style_repr = x.mean(dim=1)
        
        # Classify to prototype
        logits = self.classifier(style_repr)
        
        return {
            'prototype_logits': logits,
            'style_representation': style_repr
        }

def create_maia_style_model(vocab_size=4096, max_players=10000):
    """Factory function to create MaiaStyle model"""
    return MaiaStyleModel(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_layers=4,
        max_individual_players=max_players,
        engine_features=5
    )

def create_prototype_matcher(num_prototypes=200):
    """Factory function to create prototype matching network"""
    return PrototypeMatchingNetwork(
        d_model=256,
        num_prototypes=num_prototypes
    )

# Test the model
if __name__ == "__main__":
    print("🧠 Testing MaiaStyle Model...")
    
    model = create_maia_style_model()
    matcher = create_prototype_matcher()
    
    # Test inputs
    batch_size = 4
    seq_len = 20
    
    move_sequence = torch.randint(0, 4096, (batch_size, seq_len))
    population_id = torch.randint(0, 11, (batch_size,))
    individual_id = torch.randint(0, 100, (batch_size,))
    engine_features = torch.randn(batch_size, 5)
    
    # Forward pass
    with torch.no_grad():
        output = model(move_sequence, population_id, individual_id, engine_features)
        
        print(f"✓ Hybrid logits shape: {output['hybrid_logits'].shape}")
        print(f"✓ Mixing weights: {output['mixing_weights'][0]}")
        print(f"✓ Individual embedding shape: {output['individual_embedding'].shape}")
    
    print("🎯 Testing Prototype Matcher...")
    
    move_features = torch.randn(batch_size, seq_len, 256)
    with torch.no_grad():
        match_output = matcher(move_features)
        print(f"✓ Prototype logits shape: {match_output['prototype_logits'].shape}")
    
    print("🎉 MaiaStyle architecture test complete!")
