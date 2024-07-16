import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Define transforms for preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load data from folders
train_data = datasets.MNIST('/home/shirisha-ranganamaina/Downloads/archive/flowers/daisy/train', download=True, train=True, transform=transform)
test_data = datasets.MNIST('/home/shirisha-ranganamaina/Downloads/archive/flowers/daisy/test', download=True, train=False, transform=transform)
validation_data = datasets.MNIST('/home/shirisha-ranganamaina/Downloads/archive/flowers/daisy/validation', download=True, train=False, transform=transform)

# Data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=False)

# Multi-layer perceptron (MLP) model
mlp = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mlp.parameters(), lr=0.01, momentum=0.9)

# Train the network
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.view(-1, 784)
        optimizer.zero_grad()
        outputs = mlp(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / (i + 1):.4f}')

# Evaluate on training set
train_correct = 0
train_total = 0
with torch.no_grad():
    for data in train_loader:
        inputs, labels = data
        inputs = inputs.view(-1, 784)
        outputs = mlp(inputs)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
print(f'Training Accuracy: {100 * train_correct / train_total:.2f} %')

# Evaluate on test set
test_correct = 0
test_total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs = inputs.view(-1, 784)
        outputs = mlp(inputs)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
print(f'Test Accuracy: {100 * test_correct / test_total:.2f} %')

# Evaluate on validation set
val_correct = 0
val_total = 0
with torch.no_grad():
    for data in validation_loader:
        inputs, labels = data
        inputs = inputs.view(-1, 784)
        outputs = mlp(inputs)
        _, predicted = torch.max(outputs.data, 1)
        val_total += labels.size(0)
        val_correct += (predicted == labels).sum().item()
print(f'Validation Accuracy: {100 * val_correct / val_total:.2f} %')
