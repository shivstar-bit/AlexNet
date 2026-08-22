import torch

def train_model(model, train_loader, validation_loader, criterion, optimizer, scheduler, num_epochs=30, device='cpu'):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * images.size(0)
            _, predict = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predict == labels).sum().item()

        train_loss = running_train_loss / len(train_loader.dataset)
        train_acc = correct_train / total_train

        model.eval()
        running_validation_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)  
                loss = criterion(outputs, labels)     
                running_validation_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item() 

        val_loss = running_validation_loss / len(validation_loader.dataset)
        val_acc = 100 * correct_val / total_val   
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_loss)

    return model
