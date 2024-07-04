import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Image size
img_size = 50

# Data preparation
def load_images(folder, label):
    data = []
    for filename in os.listdir(folder):
        try:
            path = os.path.join(folder, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size))
            img_array = np.array(img)
            data.append([img_array, np.array(label)])
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            pass
    return data

ben_training_folder = "melanoma_cancer_dataset/train/benign/"
mal_training_folder = "melanoma_cancer_dataset/train/malignant/"
ben_testing_folder = "melanoma_cancer_dataset/test/benign/"
mal_testing_folder = "melanoma_cancer_dataset/test/malignant/"

ben_training_data = load_images(ben_training_folder, [1, 0])
mal_training_data = load_images(mal_training_folder, [0, 1])
ben_testing_data = load_images(ben_testing_folder, [1, 0])
mal_testing_data = load_images(mal_testing_folder, [0, 1])

# Balance the training data
ben_training_data = ben_training_data[:len(mal_training_data)]

print(f"Benign training count: {len(ben_training_data)}")
print(f"Malignant training count: {len(mal_training_data)}")
print(f"Benign testing count: {len(ben_testing_data)}")
print(f"Malignant testing count: {len(mal_testing_data)}")

# Combine and shuffle data
training_data = ben_training_data + mal_training_data
np.random.shuffle(training_data)
testing_data = ben_testing_data + mal_testing_data
np.random.shuffle(testing_data)

# Separate images and labels
training_images = np.array([item[0] for item in training_data])
training_labels = np.array([item[1] for item in training_data])
testing_images = np.array([item[0] for item in testing_data])
testing_labels = np.array([item[1] for item in testing_data])

# Save the combined data
np.save("melanoma_training_images.npy", training_images)
np.save("melanoma_training_labels.npy", training_labels)
np.save("melanoma_testing_images.npy", testing_images)
np.save("melanoma_testing_labels.npy", testing_labels)

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
        self.fc1 = nn.Linear(128 * 2 * 2, 512)  # Update this to actual input size
        self.fc2 = nn.Linear(512, 2)  # Binary classification

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = x.view(-1, 128 * 2 * 2)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load training images and labels
train_images = np.load("melanoma_training_images.npy")
train_labels = np.load("melanoma_training_labels.npy")

# Normalize images
train_X = torch.Tensor(train_images).view(-1, 1, img_size, img_size) / 255.0

# Convert labels to tensor
train_y = torch.Tensor(train_labels)

# Initialize the neural network
net = Net()

# Define the optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Define the loss function
loss_function = nn.CrossEntropyLoss()

# Batch size and number of epochs
batch_size = 100
epochs = 2

# Training loop
for epoch in range(epochs):
    for i in range(0, len(train_X), batch_size):
        print(f"EPOCH {epoch+1}, fraction complete: {i/len(train_X):.2f}")

        batch_X = train_X[i:i+batch_size]
        batch_y = train_y[i:i+batch_size]

        optimizer.zero_grad()

        outputs = net(batch_X)
        loss = loss_function(outputs, torch.max(batch_y, 1)[1])  # Convert one-hot to class index
        loss.backward()
        optimizer.step()

# Save the trained model
torch.save(net.state_dict(), "saved_model.pth")

# Load testing images and labels
test_images = np.load("melanoma_testing_images.npy")
test_labels = np.load("melanoma_testing_labels.npy")

# Normalize images
test_X = torch.Tensor(test_images).view(-1, 1, img_size, img_size) / 255.0

# Convert labels to tensor
test_y = torch.Tensor(test_labels)

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():
    for i in range(len(test_X)):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1, 1, img_size, img_size))[0]
        predicted_class = torch.argmax(net_out)
        if predicted_class == real_class:
            correct += 1
        total += 1

print(f"Accuracy: {correct / total:.2f}")
