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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
import warnings
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# Configuration
config = {
    "dataset_folder": "Dataset/ShapeNetCoreV2/obj-ShapeNetCoreV2",
    "label_file": "Dataset/ShapeNetCoreV2/label/Final_Validated_Regularity_Levels.xlsx",
    "num_points": 2048,
    "batch_size": 16,
    "num_epochs": 70,  # Reduced epochs for debugging
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_classes": 4,  # Updated to 4 classes (1-4)
    "sample_size": None,  # Limit dataset size for debugging
}

# Load 3D mesh
def load_mesh(file_path, num_points):
    try:
        mesh = trimesh.load(file_path, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("Invalid mesh format")
        vertices = mesh.vertices
        normals = getattr(mesh, "vertex_normals", None)
        if normals is None:
            raise ValueError("Mesh does not have vertex normals")

        if len(vertices) > num_points:
            indices = np.random.choice(len(vertices), num_points, replace=False)
            vertices, normals = vertices[indices], normals[indices]
        elif len(vertices) < num_points:
            padding = num_points - len(vertices)
            vertices = np.vstack([vertices, np.zeros((padding, 3))])
            normals = np.vstack([normals, np.zeros((padding, 3))])
        return vertices, normals
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

# Data augmentation
def augment_mesh(vertices, normals):
    rotation_matrix = trimesh.transformations.random_rotation_matrix()[:3, :3]
    vertices = np.dot(vertices, rotation_matrix.T)
    jitter = np.random.normal(0, 0.02, vertices.shape)
    vertices += jitter
    return vertices, normals

# Dataset class
class MeshDataset(Dataset):
    def __init__(self, vertices_list, normals_list, targets, num_points):
        self.vertices_list = vertices_list
        self.normals_list = normals_list
        self.targets = targets
        self.num_points = num_points

    def __len__(self):
        return len(self.vertices_list)

    def __getitem__(self, idx):
        vertices = self.vertices_list[idx]
        normals = self.normals_list[idx]
        label = self.targets[idx]

        vertices = torch.tensor(vertices, dtype=torch.float32)
        normals = torch.tensor(normals, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return {"vertices": vertices, "normals": normals}, label

# MeshNet model
class MeshNet(nn.Module):
    def __init__(self, num_classes):
        super(MeshNet, self).__init__()
        self.conv1 = nn.Conv1d(6, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, vertices, normals):
        x = torch.cat([vertices, normals], dim=2).transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = torch.max(x, dim=-1)[0]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Preprocess data
def preprocess_data(config):
    label_data = pd.read_excel(config["label_file"])
    label_data = label_data[["Object ID (Dataset Original Object ID)", "Final Inside Level", "Folder Name"]]
    label_data.dropna(inplace=True)

    # Filter invalid labels
    label_data = label_data[label_data["Final Inside Level"].between(1, config["num_classes"])]

    if config["sample_size"] and config["sample_size"] < len(label_data):
        label_data = label_data.sample(n=config["sample_size"], random_state=42)

    vertices_list, normals_list, targets = [], [], []

    for _, row in tqdm(label_data.iterrows(), total=len(label_data)):
        object_id = row["Object ID (Dataset Original Object ID)"]
        folder_name = str(int(row["Folder Name"])).zfill(8)
        label = int(row["Final Inside Level"]) - 1  # Convert to zero-based indexing

        file_path = os.path.join(config["dataset_folder"], folder_name, object_id.strip(), "models", "model_normalized.obj")
        vertices, normals = load_mesh(file_path, config["num_points"])
        if vertices is not None:
            vertices, normals = augment_mesh(vertices, normals)
            vertices_list.append(vertices)
            normals_list.append(normals)
            targets.append(label)

    # Validate processed labels
    assert all(0 <= label < config["num_classes"] for label in targets), "Found invalid labels."

    return vertices_list, normals_list, np.array(targets)

# Training and evaluation
def train_model(config, vertices_list, normals_list, targets):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    device = config["device"]

    for fold, (train_idx, test_idx) in enumerate(skf.split(vertices_list, targets)):
        print(f"Fold {fold + 1}")
        train_dataset = MeshDataset(
            [vertices_list[i] for i in train_idx],
            [normals_list[i] for i in train_idx],
            targets[train_idx],
            num_points=config["num_points"]
        )
        test_dataset = MeshDataset(
            [vertices_list[i] for i in test_idx],
            [normals_list[i] for i in test_idx],
            targets[test_idx],
            num_points=config["num_points"]
        )

        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

        model = MeshNet(config["num_classes"]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        for epoch in range(config["num_epochs"]):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                inputs, labels = batch
                vertices = inputs["vertices"].to(device)
                normals = inputs["normals"].to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(vertices, normals)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            print(f"Epoch {epoch + 1}/{config['num_epochs']}, Loss: {total_loss / len(train_loader):.4f}")

        evaluate_model(model, test_loader)

def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            vertices = inputs["vertices"].to(config["device"])
            normals = inputs["normals"].to(config["device"])
            outputs = model(vertices, normals)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Main
if __name__ == "__main__":
    vertices_list, normals_list, targets = preprocess_data(config)
    train_model(config, vertices_list, normals_list, targets)
