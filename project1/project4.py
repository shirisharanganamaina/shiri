import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

# Define paths
train_path = '/home/shirisha-ranganamaina/Downloads/archive/flowers/daisy/train'
val_path = '/home/shirisha-ranganamaina/Downloads/archive/flowers/daisy/validation'
test_path = '/home/shirisha-ranganamaina/Downloads/archive/flowers/daisy/test'

# Transform for data augmentation
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# Load datasets
images_list = []
labels_list = []
paths = [train_path, val_path, test_path]
for path in paths:
    images = []
    labels = []
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(path, filename)).convert('RGB')
            img = transform(img)
            img_array = img.numpy()
            images.append(img_array)
            labels.append(1 if filename.startswith('flower') else 0)
    images_list.append(images)
    labels_list.append(labels)

# Create datasets
train_dataset = list(zip(images_list[0], labels_list[0]))
val_dataset = list(zip(images_list[1], labels_list[1]))
test_dataset = list(zip(images_list[2], labels_list[2]))

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define a simple CNN
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 8 * 8, 128),
    nn.Dropout(0.5),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

# Regularization techniques to test
regularization_techniques = {
    'Dropout (0.5)': (0.5, 0.0),
    'L2 Regularization': (0.1, 0.0),
    'L1 Regularization': (0.01, 0.0),
}

# Different epochs to test
epochs_list = [5]

for technique, (dropout_rate, weight_decay) in regularization_techniques.items():
    print(f'\nTraining with {technique}...\n')
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 128),
        nn.Dropout(dropout_rate),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=weight_decay)
    for epochs in epochs_list:
        print(f'Training for {epochs} epochs...\n')
        # Training loop
        for epoch in range(epochs):
            running_loss = 0.0
            model.train()
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels.float())
                # L1 regularization
                if weight_decay > 0:
                    l1_norm = sum(p.abs().sum() for p in model.parameters())
                    loss += weight_decay * l1_norm
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}')

        # Evaluate on training set
        train_correct = 0
        train_total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in train_loader:
                outputs = model(images)
                predicted = (outputs.squeeze() > 0.5).float()
                train_total += labels.size(0)
                train_correct += (predicted == labels.float()).sum().item()
        train_accuracy = 100 * train_correct / train_total
        print(f'Training Accuracy: {train_accuracy:.2f} %')

        # Evaluate on validation set
        val_correct = 0
        val_total = 0
        val_loss = 0.01
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels.float())
                val_loss += loss.item()
                predicted = (outputs.squeeze() > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels.float()).sum().item()
        val_accuracy = 100 * val_correct / val_total
        print(f'Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f} %')

        # Evaluate on test set
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                predicted = (outputs.squeeze() > 0.5).float()
                test_total += labels.size(0)
                test_correct += (predicted == labels.float()).sum().item()
        test_accuracy = 100 * test_correct / test_total
        print(f'Test Accuracy: {test_accuracy:.2f} %')
