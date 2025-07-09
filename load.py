import torch
import torch.nn as nn
import decoder_model
import os

input_dim = 768
hidden_size = 768
num_layers = 2
vocab_size = 30522
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FOLDER_PATH = "./model_data/"

# Create folder if it doesn't exist
os.makedirs(FOLDER_PATH, exist_ok=True)

def initializeModel():  # for creating a fresh model to train from scratch
    decoder = decoder_model.MultiPerspectiveNN(input_dim, hidden_size, num_layers)
    return decoder.to(device)

def initializeOptimizer(model, lr=0.001):  # initialize using adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return optimizer

def saveModel(model):  # this gets modified later
    PATH = f"{FOLDER_PATH}model_state_dict.pth"
    torch.save(model.state_dict(), PATH)
    print("Saved model")

def load_saved_model(model):  # placeholder loader
    state_dict = torch.load(f"{FOLDER_PATH}model_state_dict.pth", map_location=device)
    model.load_state_dict(state_dict)
    print("Loaded model")
    return model.to(device)

def save_optimizer(optimizer, filepath=f"{FOLDER_PATH}optimizer_state_dict.pth"):
    """Save optimizer state dict"""
    torch.save(optimizer.state_dict(), filepath)
    print(f"Saved optimizer state to {filepath}")

def load_optimizer(model, lr=0.001):  # recreate optimizer for loaded model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("Loaded optimizer")
    return optimizer

def save_checkpoint(model, optimizer, epoch, batch_idx, filepath=f"{FOLDER_PATH}checkpoint.pth"):
    """Save complete training checkpoint with epoch and batch index"""
    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
      #  'loss': loss, removed this for now, could add later for more functionality
    }
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint at epoch {epoch}, batch {batch_idx}")

def load_checkpoint(model, optimizer, filepath=f"{FOLDER_PATH}checkpoint.pth"):
    """Load complete training checkpoint, returning resume indices"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model = model.to(device)
    epoch = checkpoint.get('epoch', 0)
    batch_idx = checkpoint.get('batch_idx', 0)
    # loss = checkpoint.get('loss', None), can be added later
    print(f"Loaded checkpoint from epoch {epoch}, batch {batch_idx}")
    return model, optimizer, epoch, batch_idx # loss

def save_lstm_cell_state(cell_state, filepath=f"{FOLDER_PATH}lstm_cell_state.pth"):
    """Save LSTM cell state (c_t) only"""
    torch.save(cell_state, filepath)
    print(f"Saved LSTM cell state to {filepath}")

def load_lstm_cell_state(filepath=f"{FOLDER_PATH}lstm_cell_state.pth"):
    """Load LSTM cell state (c_t) only"""
    cell_state = torch.load(filepath, map_location=device)
    print(f"Loaded LSTM cell state from {filepath}")
    return cell_state

# Example usage:
if __name__ == "__main__":
    # Initialize or load as needed
    model = initializeModel()
    optimizer = initializeOptimizer(model)
    # To resume:
    # model, optimizer, start_epoch, start_batch, _ = load_checkpoint(model, optimizer)
    print(f"Model is on device: {next(model.parameters()).device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Saving to folder: {FOLDER_PATH}")
