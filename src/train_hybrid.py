import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import json
from src.hybrid_model import HybridChessModel, ChessEngineIntegration, HybridTrainer

class HybridChessDataset(Dataset):
    def __init__(self, data, vocab_size=4096):
        self.data = data
        self.vocab_size = vocab_size
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'sequence': torch.tensor(item['sequence'], dtype=torch.long),
            'target': torch.tensor(item['target'], dtype=torch.long),
            'rating_bin': torch.tensor(item['rating_bin'], dtype=torch.long),
            'engine_features': torch.tensor(item['engine_features'], dtype=torch.float32)
        }

def train_hybrid_model(data_path, model_save_path, epochs=50, batch_size=32, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize engine (adjust path for AI Panther cluster)
    engine_path = "/usr/games/stockfish"  # Common Linux path
    try:
        engine = ChessEngineIntegration(engine_path=engine_path, depth=8)
        print("Stockfish engine loaded successfully")
    except:
        print("Warning: Stockfish not found, using dummy engine features")
        engine = None
    
    # Load and prepare data
    print("Loading training data...")
    with open(data_path, 'r') as f:
        raw_data = json.load(f)
    
    # Create hybrid model and trainer
    model = HybridChessModel(vocab_size=4096, rating_bins=10).to(device)
    
    if engine:
        trainer = HybridTrainer(model, engine, device)
        enhanced_data = trainer.prepare_training_data(raw_data)
    else:
        # Fallback without engine
        enhanced_data = []
        for game in raw_data:
            for i, move in enumerate(game['moves'][:-1]):
                enhanced_data.append({
                    'sequence': game['moves'][:i+1],
                    'target': game['moves'][i+1],
                    'rating_bin': min(9, max(0, (game['rating'] - 800) // 200)),
                    'engine_features': np.zeros(5)  # Dummy features
                })
    
    # Create dataset and dataloader
    dataset = HybridChessDataset(enhanced_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Loss functions
    style_criterion = nn.CrossEntropyLoss()
    engine_criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    print(f"Training hybrid model for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        style_loss_sum = 0
        engine_loss_sum = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            optimizer.zero_grad()
            
            # Move to device
            sequences = batch['sequence'].to(device)
            targets = batch['target'].to(device)
            rating_bins = batch['rating_bin'].to(device)
            engine_features = batch['engine_features'].to(device)
            
            # Forward pass
            outputs = model(sequences, rating_bins, engine_features)
            
            # Multi-objective loss
            style_loss = style_criterion(outputs['style_logits'], targets)
            engine_loss = engine_criterion(outputs['engine_logits'], targets)
            
            # Weighted combination
            total_batch_loss = 0.7 * style_loss + 0.3 * engine_loss
            
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += total_batch_loss.item()
            style_loss_sum += style_loss.item()
            engine_loss_sum += engine_loss.item()
            
            pbar.set_postfix({
                'Loss': f"{total_batch_loss.item():.4f}",
                'Style': f"{style_loss.item():.4f}",
                'Engine': f"{engine_loss.item():.4f}"
            })
        
        scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        avg_style = style_loss_sum / len(dataloader)
        avg_engine = engine_loss_sum / len(dataloader)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Style={avg_style:.4f}, Engine={avg_engine:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"{model_save_path}_epoch_{epoch+1}.pt")
    
    # Save final model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train_hybrid_model(
        data_path="data/processed/training_data.json",
        model_save_path="models/hybrid_chess_model.pt",
        epochs=50,
        batch_size=64,
        lr=1e-4
    )
