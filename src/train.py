import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from tqdm import tqdm
import json

from model import create_model
from data_utils import MoveTokenizer, ChessDataset, load_pgn_files, create_player_mapping

def train_model(data_dir, model_save_dir, epochs=50, batch_size=32, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    pgn_files = load_pgn_files(data_dir)
    print(f"Found {len(pgn_files)} PGN files")
    
    # Create tokenizer and player mapping
    tokenizer = MoveTokenizer()
    tokenizer.build_vocab(pgn_files)
    player_mapping = create_player_mapping(pgn_files)
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Players: {list(player_mapping.keys())}")
    
    # Create dataset
    dataset = ChessDataset(pgn_files, tokenizer, player_mapping)
    print(f"Total training examples: {len(dataset)}")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = create_model(
        num_players=len(player_mapping),
        vocab_size=tokenizer.vocab_size
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            sequences = batch['sequence'].to(device)
            targets = batch['target'].to(device)
            player_ids = batch['player_id'].to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences, player_ids)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch['sequence'].to(device)
                targets = batch['target'].to(device)
                player_ids = batch['player_id'].to(device)
                
                outputs = model(sequences, player_ids)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        # Calculate metrics
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'tokenizer': tokenizer,
                'player_mapping': player_mapping,
                'vocab_size': tokenizer.vocab_size,
                'num_players': len(player_mapping)
            }, os.path.join(model_save_dir, 'best_model.pth'))
        
        scheduler.step()
    
    print("Training completed!")

if __name__ == "__main__":
    train_model(
        data_dir="../data/raw_games",
        model_save_dir="../models",
        epochs=50,
        batch_size=32,
        lr=1e-4
    )
