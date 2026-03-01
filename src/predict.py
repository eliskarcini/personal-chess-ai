import torch
import chess
from model import create_model

class ChessPredictor:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and metadata
        checkpoint = torch.load(model_path, map_location=self.device)
        self.tokenizer = checkpoint['tokenizer']
        self.player_mapping = checkpoint['player_mapping']
        
        # Create and load model
        self.model = create_model(
            num_players=checkpoint['num_players'],
            vocab_size=checkpoint['vocab_size']
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Reverse player mapping for lookup
        self.id_to_player = {v: k for k, v in self.player_mapping.items()}
    
    def predict_next_move(self, move_sequence, player_name, top_k=5):
        """
        Predict next move given sequence and player
        
        Args:
            move_sequence: List of moves in string format ['e4', 'e5', 'Nf3']
            player_name: Name of the player to predict for
            top_k: Return top k predictions
        
        Returns:
            List of (move, probability) tuples
        """
        if player_name not in self.player_mapping:
            raise ValueError(f"Unknown player: {player_name}")
        
        # Encode sequence
        encoded_seq = self.tokenizer.encode(move_sequence)
        
        # Pad sequence (same as training)
        max_seq_len = 50
        if len(encoded_seq) > max_seq_len:
            encoded_seq = encoded_seq[-max_seq_len:]
        else:
            encoded_seq = [0] * (max_seq_len - len(encoded_seq)) + encoded_seq
        
        # Convert to tensors
        sequence_tensor = torch.tensor([encoded_seq], dtype=torch.long).to(self.device)
        player_tensor = torch.tensor([self.player_mapping[player_name]], dtype=torch.long).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(sequence_tensor, player_tensor)
            probabilities = torch.softmax(outputs, dim=-1)
        
        # Get top k predictions
        top_probs, top_indices = torch.topk(probabilities[0], top_k)
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            move = self.tokenizer.id_to_move.get(idx.item(), '<UNK>')
            if move not in ['<PAD>', '<START>', '<END>', '<UNK>']:
                predictions.append((move, prob.item()))
        
        return predictions
    
    def evaluate_game(self, moves, white_player, black_player):
        """
        Evaluate model accuracy on a complete game
        
        Args:
            moves: List of all moves in the game
            white_player: Name of white player
            black_player: Name of black player
        
        Returns:
            Dictionary with accuracy statistics
        """
        correct_predictions = 0
        total_predictions = 0
        
        for i in range(1, len(moves)):
            current_sequence = moves[:i]
            actual_move = moves[i]
            current_player = white_player if i % 2 == 1 else black_player
            
            if current_player in self.player_mapping:
                predictions = self.predict_next_move(current_sequence, current_player, top_k=1)
                if predictions and predictions[0][0] == actual_move:
                    correct_predictions += 1
                total_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        return {
            'accuracy': accuracy,
            'correct': correct_predictions,
            'total': total_predictions
        }

def main():
    # Example usage
    predictor = ChessPredictor("../models/best_model.pth")
    
    # Predict next move
    moves = ["e4", "e5", "Nf3"]
    player = "Player_A"  # Replace with actual player name
    
    predictions = predictor.predict_next_move(moves, player, top_k=3)
    
    print(f"Top 3 predictions for {player}:")
    for move, prob in predictions:
        print(f"  {move}: {prob:.3f}")

if __name__ == "__main__":
    main()
