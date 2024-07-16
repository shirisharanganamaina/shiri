import os
from PIL import Image
import numpy as np

# Define paths
train_path = '/home/shirisha-ranganamaina/Downloads/archive/flowers/daisy/train'
validation_path = '/home/shirisha-ranganamaina/Downloads/archive/flowers/daisy/validation'
test_path = '/home/shirisha-ranganamaina/Downloads/archive/flowers/daisy/test'

# Load training data
train_images = []
train_labels = []
for filename in os.listdir(train_path):
    if filename.endswith('.jpg'):  # Assuming images are in JPG format
        img = Image.open(os.path.join(train_path, filename))
        img = img.resize((32, 32))  # Resize image to a standard size
        img_array = np.array(img)
        train_images.append(img_array.flatten() / 255.0)  # Flatten and normalize image
        train_labels.append(1 if filename.startswith('flower') else -1)  # Assuming flowers are positive class (1), others are negative class (-1)

X_train = np.array(train_images)
y_train = np.array(train_labels)

# Load validation data
val_images = []
val_labels = []
for filename in os.listdir(validation_path):
    if filename.endswith('.jpg'):  # Assuming images are in JPG format
        img = Image.open(os.path.join(validation_path, filename))
        img = img.resize((32, 32))  # Resize image to a standard size
        img_array = np.array(img)
        val_images.append(img_array.flatten() / 255.0)  # Flatten and normalize image
        val_labels.append(1 if filename.startswith('flower') else -1)  # Assuming flowers are positive class (1), others are negative class (-1)

X_val = np.array(val_images)
y_val = np.array(val_labels)

# Load test data
test_images = []
test_labels = []
for filename in os.listdir(test_path):
    if filename.endswith('.jpg'):  # Assuming images are in JPG format
        img = Image.open(os.path.join(test_path, filename))
        img = img.resize((32, 32))  # Resize image to a standard size
        img_array = np.array(img)
        test_images.append(img_array.flatten() / 255.0)  # Flatten and normalize image
        test_labels.append(1 if filename.startswith('flower') else -1)  # Assuming flowers are positive class (1), others are negative class (-1)

X_test = np.array(test_images)
y_test = np.array(test_labels)

# Initialize perceptron weights and bias
num_features = X_train.shape[1]
weights = np.zeros(num_features)
bias = 0
learning_rate = 0.1
num_epochs = 100

# Train the perceptron
for epoch in range(num_epochs):
    # Training loop
    for i in range(len(X_train)):
        prediction = np.dot(X_train[i], weights) + bias
        if prediction >= 0:
            y_pred = 1
        else:
            y_pred = -1
        if y_pred != y_train[i]:
            weights += learning_rate * y_train[i] * X_train[i]
            bias += learning_rate * y_train[i]

# Evaluate on training set
correct_train = 0
for i in range(len(X_train)):
    prediction = np.dot(X_train[i], weights) + bias
    if prediction >= 0:
        y_pred = 1
    else:
        y_pred = -1
    if y_pred == y_train[i]:
        correct_train += 1

train_accuracy = correct_train / len(X_train) * 100
print(f'Training Accuracy: {train_accuracy:.2f}%')

# Evaluate on validation set
correct_val = 0
for i in range(len(X_val)):
    prediction = np.dot(X_val[i], weights) + bias
    if prediction >= 0:
        y_pred = 1
    else:
        y_pred = -1
    if y_pred == y_val[i]:
        correct_val += 1

validation_accuracy = correct_val / len(X_val) * 100
print(f'Validation Accuracy: {validation_accuracy:.2f}%')

# Evaluate on test set
correct_test = 0
for i in range(len(X_test)):
    prediction = np.dot(X_test[i], weights) + bias
    if prediction >= 0:
        y_pred = 1
    else:
        y_pred = -1
    if y_pred == y_test[i]:
        correct_test += 1

test_accuracy = correct_test / len(X_test) * 100
print(f'Test Accuracy: {test_accuracy:.2f}%')
