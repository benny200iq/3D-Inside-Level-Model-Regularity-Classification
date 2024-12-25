import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import trimesh
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
import random
import warnings

warnings.filterwarnings("ignore")

# Load labels from Excel file
label_file = 'Dataset/3d-future-dataset/label/Final_Validated_Regularity_Levels.xlsx'
label_data = pd.read_excel(label_file)

# Extract relevant columns
labels = label_data[['Object ID (Dataset Original Object ID)', 'Final Inside Level']]

# Set the maximum number of data points to use
MAX = len(labels)
MAX_VALUE = 2000
labels = labels[:MAX_VALUE]

# Path to the folder containing 3D objects
obj_folder = 'Dataset/3d-future-dataset/obj-3d.future'

# Feature extraction function
def extract_features_from_obj(file_path):
    try:
        mesh = trimesh.load(file_path, force='mesh')
        return mesh.vertices, mesh.faces
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

# Prepare dataset
vertices_list = []
targets = []

for index, row in tqdm(labels.iterrows(), total=len(labels)):
    object_id = row['Object ID (Dataset Original Object ID)']
    regularity_level = row['Final Inside Level']

    # Construct the path using FolderName and Object ID
    obj_file = os.path.join(obj_folder, object_id.strip(), f"normalized_model.obj")

    # Extract features
    vertices, _ = extract_features_from_obj(obj_file)
    if vertices is not None:
        vertices_list.append(vertices)
        targets.append(regularity_level)

# Ensure dataset is not empty
if len(vertices_list) == 0:
    print("No features extracted. Please check the dataset and feature extraction process.")
    exit()

y = np.array(targets) - 1  # Adjust labels to be zero-indexed

class MeshDataset(Dataset):
    def __init__(self, vertices_list, targets, num_points=2048, augment=False):
        self.vertices_list = vertices_list
        self.targets = targets
        self.num_points = num_points
        self.augment = augment

    def __len__(self):
        return len(self.vertices_list)

    def __getitem__(self, idx):
        vertices = self.vertices_list[idx]
        label = self.targets[idx]

        # Pad or sample points
        if len(vertices) > self.num_points:
            vertices = vertices[:self.num_points]
        elif len(vertices) < self.num_points:
            padding = self.num_points - len(vertices)
            vertices = np.pad(vertices, ((0, padding), (0, 0)), 'constant')

        # Data augmentation (random scaling, rotation, jitter)
        if self.augment:
            vertices = self._augment(vertices)

        vertices = torch.tensor(vertices.T, dtype=torch.float32)  # Transpose to (3, num_points)
        label = torch.tensor(label, dtype=torch.long)
        return vertices, label

    def _augment(self, vertices):
        # Random scaling
        scale = random.uniform(0.8, 1.2)
        vertices *= scale

        # Random rotation
        angle = random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        vertices = np.dot(vertices, rotation_matrix)

        # Random jitter
        jitter = np.random.normal(0, 0.02, vertices.shape)
        vertices += jitter

        return vertices

# Split dataset into training and testing sets
split_idx = int(0.8 * len(vertices_list))
train_vertices = vertices_list[:split_idx]
test_vertices = vertices_list[split_idx:]
train_targets = y[:split_idx]
test_targets = y[split_idx:]

train_dataset = MeshDataset(train_vertices, train_targets, num_points=2048, augment=True)
test_dataset = MeshDataset(test_vertices, test_targets, num_points=2048, augment=False)

# Define MeshNet model
class MeshNet(nn.Module):
    def __init__(self, num_classes):
        super(MeshNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn_fc1 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]  # Global max pooling
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Training
    model = MeshNet(num_classes=len(np.unique(y))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 regularization
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for vertices, labels in train_loader:
            vertices, labels = vertices.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(vertices)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Evaluate model
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for vertices, labels in test_loader:
            vertices, labels = vertices.to(device), labels.to(device)
            outputs = model(vertices)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(len(np.unique(y))), yticklabels=np.arange(len(np.unique(y))))
    plt.title('MeshNet Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
