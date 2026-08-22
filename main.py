import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import get_dataloaders
from src.model import AlexNet
from src.train import train_model

def main():
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading datasets...")
    train_loader, validation_loader = get_dataloaders(batch_size=64, num_workers=2, data_dir='./data')

    # 2. Initialize Model, Loss, Optimizer, and Scheduler
    print("Initializing model...")
    model = AlexNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # 3. Train Model
    num_epochs = 30
    print(f"Starting training for {num_epochs} epochs...")
    train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=device
    )
    
    # 4. Save Model
    model_save_path = "alexnet_cifar10.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    print("Training complete!")

if __name__ == "__main__":
    main()