import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import time
from tqdm import tqdm
from resnet import resnet18gai

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load data
train_data = ImageFolder('./cropped/train', transform=train_transform)
val_data = ImageFolder('./cropped/test', transform=val_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=2)

# Load pre-trained ResNet-18 model
model = resnet18gai().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr =0.001,momentum=0.9,nesterov=True, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 70], gamma=0.05)

# Train the model
num_epochs = 50
best_acc = 0.0
for epoch in range(num_epochs):
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    start_time = time.time()

    # Train
    model.train()
    with tqdm(train_loader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in tepoch:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            tepoch.set_postfix(loss=train_loss/train_total, acc=train_correct/train_total)

    # Validate
    model.eval()
    with torch.no_grad():
        with tqdm(val_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                tepoch.set_postfix(loss=val_loss/val_total, acc=val_correct/val_total)

    lr_scheduler.step()
    end_time = time.time()
    print(f"Epoch {epoch+1}/{num_epochs} took {end_time - start_time:.2f} seconds")
    print(f"Train loss: {train_loss/train_total:.4f}, Train accuracy: {train_correct/train_total:.4f}")
    print(f"Validation loss: {val_loss/val_total:.4f}, Validation accuracy: {val_correct/val_total:.4f}")
    # Update best accuracy
    if val_correct / val_total > best_acc:
        best_acc = val_correct / val_total

print(f"Best validation accuracy: {best_acc:.4f}")

