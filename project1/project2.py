import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

# Load datasets
train_path = '/home/shirisha-ranganamaina/Downloads/archive/flowers/daisy/train'
val_path = '/home/shirisha-ranganamaina/Downloads/archive/flowers/daisy/validation'
test_path = '/home/shirisha-ranganamaina/Downloads/archive/flowers/daisy/test'

train_images, train_labels = [], []
val_images, val_labels = [], []
test_images, test_labels = [], []

# Load training data
for filename in os.listdir(train_path):
    if filename.endswith('.jpg'):
        img = Image.open(os.path.join(train_path, filename)).resize((32, 32))
        img_array = np.array(img).astype('float32') / 255.0
        train_images.append(img_array)
        train_labels.append(1 if filename.startswith('flower') else 0)

# Load validation data
for filename in os.listdir(val_path):
    if filename.endswith('.jpg'):
        img = Image.open(os.path.join(val_path, filename)).resize((32, 32))
        img_array = np.array(img).astype('float32') / 255.0
        val_images.append(img_array)
        val_labels.append(1 if filename.startswith('flower') else 0)

# Load testing data
for filename in os.listdir(test_path):
    if filename.endswith('.jpg'):
        img = Image.open(os.path.join(test_path, filename)).resize((32, 32))
        img_array = np.array(img).astype('float32') / 255.0
        test_images.append(img_array)
        test_labels.append(1 if filename.startswith('flower') else 0)

train_dataset = list(zip(train_images, train_labels))
val_dataset = list(zip(val_images, val_labels))
test_dataset = list(zip(test_images, test_labels))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define a simple neural network
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(32 * 32 * 3, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            val_loss += criterion(outputs.squeeze(), labels.float()).item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss / len(val_loader):.4f}')

# Evaluate on training set
train_correct = 0
train_total = 0
for images, labels in train_loader:
    outputs = model(images)
    predicted = (outputs.squeeze() > 0.5).float()
    train_total += labels.size(0)
    train_correct += (predicted == labels.float()).sum().item()
train_accuracy = train_correct / train_total * 100
print(f'Training Accuracy: {train_accuracy:.2f}%')

# Evaluate on validation set
val_correct = 0
val_total = 0
for images, labels in val_loader:
    outputs = model(images)
    predicted = (outputs.squeeze() > 0.5).float()
    val_total += labels.size(0)
    val_correct += (predicted == labels.float()).sum().item()
val_accuracy = val_correct / val_total * 100
print(f'Validation Accuracy: {val_accuracy:.2f}%')

# Evaluate on test set
test_correct = 0
test_total = 0
for images, labels in test_loader:
    outputs = model(images)
    predicted = (outputs.squeeze() > 0.5).float()
    test_total += labels.size(0)
    test_correct += (predicted == labels.float()).sum().item()
test_accuracy = test_correct / test_total * 100
print(f'Test Accuracy: {test_accuracy:.2f}%')
