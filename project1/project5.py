import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
import os

# Define paths to datasets
training_path = '/home/shirisha-ranganamaina/Downloads/archive/flowers/daisy/train'
validation_path = '/home/shirisha-ranganamaina/Downloads/archive/flowers/daisy/validation'
testing_path = '/home/shirisha-ranganamaina/Downloads/archive/flowers/daisy/test'

 

# Define transforms for preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Paths to datasets
datasets = [
    {'name': 'train', 'path': training_path},
    {'name': 'validation', 'path': validation_path},
    {'name': 'test', 'path': testing_path}
]

# Initialize datasets and dataloaders
data_loaders = {}
for dataset in datasets:
    images = [f for f in os.listdir(dataset['path']) if f.endswith('.jpg') or f.endswith('.png')]
    data = []
    labels = []
    for img_name in images:
        img_path = os.path.join(dataset['path'], img_name)
        image = Image.open(img_path)
        image = transform(image)
        data.append(image)
        labels.append(torch.tensor(0))  # Assuming all labels are 0 for simplicity

    # Convert lists to tensors
    data_tensor = torch.stack(data)
    labels_tensor = torch.tensor(labels)

    # Create TensorDataset and DataLoader
    data_loaders[dataset['name']] = DataLoader(TensorDataset(data_tensor, labels_tensor), batch_size=64, shuffle=dataset['name'] == 'train')

# Neural network model
net = nn.Sequential(
    nn.Linear(224 * 224 * 3, 128),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# Loss function
criterion = nn.CrossEntropyLoss()

# List of optimization algorithms to compare
optimizers = [
    {'name': 'SGD', 'optimizer': optim.SGD(net.parameters(), lr=0.01, momentum=0.9)},
    {'name': 'Adam', 'optimizer': optim.Adam(net.parameters(), lr=0.001)},
    {'name': 'RMSprop', 'optimizer': optim.RMSprop(net.parameters(), lr=0.001)},
    {'name': 'Adagrad', 'optimizer': optim.Adagrad(net.parameters(), lr=0.01)},
    {'name': 'Adadelta', 'optimizer': optim.Adadelta(net.parameters(), lr=1.0)}
]

# Training loop
for optimizer_info in optimizers:
    optimizer_name = optimizer_info['name']
    optimizer = optimizer_info['optimizer']
    print(f"Training with {optimizer_name} optimizer...")

    # Reset model parameters
    for module in net.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Dropout):
            module.p = 0.5

    # Train the network
    num_epochs = 5
    for epoch in range(num_epochs):
        running_loss = 0.0
        net.train()  # Set the model to train mode
        for dataset_name in ['train']:
            for i, (inputs, labels) in enumerate(data_loaders[dataset_name], 0):
                optimizer.zero_grad()
                inputs = inputs.view(-1, 224 * 224 * 3)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        # Calculate training accuracy
        net.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in data_loaders['train']:
                inputs = inputs.view(-1, 224 * 224 * 3)
                outputs = net(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            train_accuracy = 100 * correct / total

            # Calculate validation accuracy
            correct = 0
            total = 0
            for inputs, labels in data_loaders['validation']:
                inputs = inputs.view(-1, 224 * 224 * 3)
                outputs = net(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            validation_accuracy = 100 * correct / total

            print(f'Epoch {epoch+1}, Loss: {running_loss / len(data_loaders["train"])}, Train Accuracy: {train_accuracy}%, Validation Accuracy: {validation_accuracy}%')

    # Evaluate on test set
    correct = 0
    total = 0
    net.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for inputs, labels in data_loaders['test']:
            inputs = inputs.view(-1, 224 * 224 * 3)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_accuracy = 100 * correct / total
    print(f'Test Accuracy with {optimizer_name} optimizer: {test_accuracy}%')