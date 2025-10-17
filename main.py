import os
import torch
from model import get_model
from data_processing import get_dataloaders
from train import train_model
from eval import evaluate_model
from configuration import Config

config = Config()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def main():
  
    # Get data loaders
    train_loader, test_loader = get_dataloaders(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    
    # Create model
    model = get_model(device)
    
    # Train model
    train_model(model, train_loader, test_loader, device)
    
    # Evaluate model
    print("\nEvaluating model...")
    report, cm = evaluate_model(model, test_loader, device)
    print("\nClassification Report:")
    print(report)

if __name__ == '__main__':
    main()