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
import warnings

warnings.filterwarnings("ignore")

# Load labels from Excel file
label_file = 'Dataset/3d-future-dataset/label/Final_Validated_Regularity_Levels.xlsx'
label_data = pd.read_excel(label_file)

# Extract relevant columns
labels = label_data[['Object ID (Dataset Original Object ID)', 'Final Inside Level']]

# Set the maximum number of data points to use
MAX = len(labels)
MAX_VALUE = MAX
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
    def __init__(self, vertices_list, targets, num_points=2048):
        self.vertices_list = vertices_list
        self.targets = targets
        self.num_points = num_points

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

        vertices = torch.tensor(vertices.T, dtype=torch.float32)  # Transpose to (3, num_points)
        label = torch.tensor(label, dtype=torch.long)
        return vertices, label

# Split dataset into training and testing sets
split_idx = int(0.8 * len(vertices_list))
train_vertices = vertices_list[:split_idx]
test_vertices = vertices_list[split_idx:]
train_targets = y[:split_idx]
test_targets = y[split_idx:]

train_dataset = MeshDataset(train_vertices, train_targets, num_points=2048)
test_dataset = MeshDataset(test_vertices, test_targets, num_points=2048)

# Define PointNet++ Model
class PointNetPlusPlus(nn.Module):
    def __init__(self, num_classes):
        super(PointNetPlusPlus, self).__init__()
        self.sa1 = self._set_abstraction_module(1024, 3, 64)
        self.sa2 = self._set_abstraction_module(512, 64, 128)
        self.sa3 = self._set_abstraction_module(256, 128, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)

    def _set_abstraction_module(self, num_points, input_dim, output_dim):
        return nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim),
            nn.Conv1d(output_dim, output_dim, 1),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim),
            nn.AdaptiveMaxPool1d(1)
        )

    def forward(self, x):
        x = self.sa1(x)
        x = self.sa2(x)
        x = self.sa3(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Training
    model = PointNetPlusPlus(num_classes=len(np.unique(y))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    num_epochs = 50
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

    from sklearn.preprocessing import OneHotEncoder

    # Initialize OneHotEncoder globally
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoder.fit(y.reshape(-1, 1))  # Fit on the full dataset

    # Initialize lists to store metrics
    accuracy_scores, precision_scores, recall_scores, f1_scores, auc_scores, log_losses = [], [], [], [], [], []

    # Cross-Validation Loop
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("\nCross-Validation Evaluation\n" + "-" * 30)
    for train_idx, test_idx in skf.split(vertices_list, y):
        train_vertices = [vertices_list[i] for i in train_idx]
        test_vertices = [vertices_list[i] for i in test_idx]
        train_targets = y[train_idx]
        test_targets = y[test_idx]

        train_dataset = MeshDataset(train_vertices, train_targets, num_points=2048)
        test_dataset = MeshDataset(test_vertices, test_targets, num_points=2048)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

        model = PointNetPlusPlus(num_classes=len(np.unique(y))).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        # Training loop for current fold
        for epoch in range(10):  # Reduced epochs for cross-validation
            model.train()
            for vertices, labels in train_loader:
                vertices, labels = vertices.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(vertices)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Evaluate model
        model.eval()
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for vertices, labels in test_loader:
                vertices, labels = vertices.to(device), labels.to(device)
                outputs = model(vertices)
                probs = torch.softmax(outputs, dim=1)  # Ensure probabilities
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

        # Transform y_true using the fitted OneHotEncoder
        y_true_encoded = one_hot_encoder.transform(np.array(y_true).reshape(-1, 1))

        # # Debugging: Print shapes to validate
        # print(f"y_true_encoded shape: {y_true_encoded.shape}")
        # print(f"y_prob shape: {np.array(y_prob).shape}")

        # Calculate log loss
        try:
            logloss = log_loss(y_true, y_prob, labels=np.unique(y))
            log_losses.append(logloss)
        except ValueError as e:
            print(f"Log Loss calculation skipped: {e}")

        # Calculate AUC-ROC
        try:
            auc = roc_auc_score(
                y_true_encoded,
                y_prob,
                multi_class='ovr',
                average='weighted'
            )
            auc_scores.append(auc)
        except ValueError as e:
            print(f"AUC-ROC calculation skipped: {e}")

        # Calculate other metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    # Print averaged metrics
    print("\nFinal Cross-Validation Metrics\n" + "-" * 40)
    print(f"Average Accuracy: {np.mean(accuracy_scores):.2f}")
    print(f"Average Precision: {np.mean(precision_scores):.2f}")
    print(f"Average Recall: {np.mean(recall_scores):.2f}")
    print(f"Average F1 Score: {np.mean(f1_scores):.2f}")
    print(f"Average AUC-ROC: {np.mean(auc_scores):.2f}")
    print(f"Average Log Loss: {np.mean(log_losses):.2f}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(len(np.unique(y))), yticklabels=np.arange(len(np.unique(y))))
    plt.title('PointNet++ Confusion Matrix [Dataset: 3d-future]')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()