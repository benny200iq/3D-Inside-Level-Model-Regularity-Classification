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
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings("ignore")

# Load labels from Excel file
label_file = 'Dataset/ShapeNetCoreV2/label/Final_Validated_Regularity_Levels.xlsx'
label_data = pd.read_excel(label_file)

# Extract relevant columns
labels = label_data[['Object ID (Dataset Original Object ID)', 'Final Inside Level' , 'Folder Name']]

# Set the maximum number of data points to use
MAX = len(label_data)
MAX_VALUE = 100

# Limit the number of data points to use
labels = labels[:MAX_VALUE]

# Path to the folder containing 3D objects
obj_folder = 'Dataset/ShapeNetCoreV2/obj-ShapeNetCoreV2'

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
    first_layer_folder = str(int(row['Folder Name'])).zfill(8)
    
    regularity_level = row['Final Inside Level']

    # Construct the path using FolderName and Object ID
    obj_file = os.path.join(obj_folder, first_layer_folder, object_id.strip(), f"models",f"model_normalized.obj")
    
    # Extract features
    vertices, _ = extract_features_from_obj(obj_file)
    if vertices is not None:
        vertices_list.append(vertices)
        targets.append(regularity_level)

# Convert to dataset format
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

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define PointNet model
class PointNet(nn.Module):
    def __init__(self, k=4):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Initialize PointNet model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PointNet(k=len(np.unique(y))).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (vertices, labels) in enumerate(train_loader):
        vertices, labels = vertices.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(vertices)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Cross-validation for evaluation
skf = StratifiedKFold(n_splits=5)
auc_scores = []
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
log_losses = []

print("\nCross-Validation Evaluation\n" + "-"*30)
for train_index, test_index in skf.split(vertices_list, y):
    train_vertices = [vertices_list[i] for i in train_index]
    test_vertices = [vertices_list[i] for i in test_index]
    train_targets = y[train_index]
    test_targets = y[test_index]

    train_dataset = MeshDataset(train_vertices, train_targets, num_points=2048)
    test_dataset = MeshDataset(test_vertices, test_targets, num_points=2048)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Re-initialize model for each fold
    model = PointNet(k=len(np.unique(y))).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    for epoch in range(10):  # Reduced epochs for cross-validation
        model.train()
        for vertices, labels in train_loader:
            vertices, labels = vertices.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(vertices)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate model
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    with torch.no_grad():
        for vertices, labels in test_loader:
            vertices, labels = vertices.to(device), labels.to(device)
            outputs = model(vertices)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_prob.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    logloss = log_loss(y_true, y_prob, labels=np.unique(y))


    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    log_losses.append(logloss)

    # Calculate AUC-ROC if applicable
    lb = LabelBinarizer()
    lb.fit(y)  # Fit on all possible labels
    y_true_binarized = lb.transform(y_true)
    if y_true_binarized.shape[1] > 1 and len(set(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true_binarized, y_prob, multi_class='ovr', average='weighted')
            auc_scores.append(auc)
        except ValueError as e:
            print(f"AUC-ROC calculation skipped: {e}")

if len(auc_scores) > 0:
    average_auc_roc = np.mean(auc_scores)
print(f"Average Accuracy: {np.mean(accuracy_scores):.2f}")
print(f"Average Precision: {np.mean(precision_scores):.2f}")
print(f"Average Recall: {np.mean(recall_scores):.2f}")
print(f"Average F1 Score: {np.mean(f1_scores):.2f}")
if len(auc_scores) > 0:
    print(f"Average AUC-ROC: {average_auc_roc:.2f}")
print(f"Average Log Loss: {np.mean(log_losses):.2f}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
conf_matrix = confusion_matrix(y_true, y_pred)
if len(conf_matrix.shape) == 2:
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
else:
    print("Confusion matrix has an unexpected shape and cannot be plotted.")
plt.title('PointNet Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()