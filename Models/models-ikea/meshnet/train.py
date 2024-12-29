# import os
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# import trimesh
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix
# )
# import seaborn as sns
# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import OneHotEncoder
# import warnings

# warnings.filterwarnings("ignore")

# # Load labels from Excel file
# label_file = 'Dataset\ikea\label\Final_Validated_Regularity_Levels.xlsx'
# label_data = pd.read_excel(label_file)

# # Extract relevant columns
# labels = label_data[['Object ID (Dataset Original Object ID)', 'Final Inside Level', 'FolderName']]

# # Set the maximum number of data points to use
# MAX = len(labels)
# labels = labels[:MAX]

# # Path to the folder containing 3D objects
# obj_folder = 'datasets/IKEA/obj-IKEA'

# # Feature extraction function
# def extract_features_from_obj(file_path):
#     try:
#         mesh = trimesh.load(file_path, force='mesh')
#         return mesh.vertices, mesh.vertex_normals
#     except Exception as e:
#         print(f"Error loading {file_path}: {e}")
#         return None, None

# # Prepare dataset
# vertices_list = []
# normals_list = []
# targets = []

# for index, row in tqdm(labels.iterrows(), total=len(labels)):
#     object_id = row['Object ID (Dataset Original Object ID)']
#     folder_id = row['FolderName']
#     regularity_level = row['Final Inside Level']

#     # Construct the path using FolderName and Object ID
#     obj_file = os.path.join(obj_folder, folder_id.strip(), object_id.strip(), "ikea_model.obj")

#     # Extract features
#     vertices, normals = extract_features_from_obj(obj_file)
#     if vertices is not None:
#         vertices_list.append(vertices)
#         normals_list.append(normals)
#         targets.append(regularity_level)

# # Ensure dataset is not empty
# if len(vertices_list) == 0:
#     print("No features extracted. Please check the dataset and feature extraction process.")
#     exit()

# y = np.array(targets) - 1  # Adjust labels to be zero-indexed

# class MeshDataset(Dataset):
#     def __init__(self, vertices_list, normals_list, targets, num_points=2048):
#         self.vertices_list = vertices_list
#         self.normals_list = normals_list
#         self.targets = targets
#         self.num_points = num_points

#     def __len__(self):
#         return len(self.vertices_list)

#     def __getitem__(self, idx):
#         vertices = self.vertices_list[idx]
#         normals = self.normals_list[idx]
#         label = self.targets[idx]

#         # Pad or sample points
#         if len(vertices) > self.num_points:
#             indices = np.random.choice(len(vertices), self.num_points, replace=False)
#             vertices, normals = vertices[indices], normals[indices]
#         elif len(vertices) < self.num_points:
#             padding = self.num_points - len(vertices)
#             vertices = np.vstack([vertices, np.zeros((padding, 3))])
#             normals = np.vstack([normals, np.zeros((padding, 3))])

#         vertices = torch.tensor(vertices, dtype=torch.float32)
#         normals = torch.tensor(normals, dtype=torch.float32)
#         label = torch.tensor(label, dtype=torch.long)
#         return {"vertices": vertices, "normals": normals}, label

# # Split dataset into training and testing sets
# split_idx = int(0.8 * len(vertices_list))
# train_vertices = vertices_list[:split_idx]
# test_vertices = vertices_list[split_idx:]
# train_normals = normals_list[:split_idx]
# test_normals = normals_list[split_idx:]
# train_targets = y[:split_idx]
# test_targets = y[split_idx:]

# train_dataset = MeshDataset(train_vertices, train_normals, train_targets, num_points=2048)
# test_dataset = MeshDataset(test_vertices, test_normals, test_targets, num_points=2048)

# # Define MeshNet Model
# class MeshNet(nn.Module):
#     def __init__(self, num_classes):
#         super(MeshNet, self).__init__()
#         self.conv1 = nn.Conv1d(6, 64, 1)
#         self.conv2 = nn.Conv1d(64, 128, 1)
#         self.conv3 = nn.Conv1d(128, 256, 1)
#         self.fc1 = nn.Linear(256, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, num_classes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=0.3)

#     def forward(self, vertices, normals):
#         x = torch.cat([vertices, normals], dim=2).transpose(1, 2)
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.relu(self.conv3(x))
#         x = torch.max(x, dim=-1)[0]  # Global Max Pooling
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # DataLoaders
#     train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
#     test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

#     # Training
#     model = MeshNet(num_classes=len(np.unique(y))).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

#     num_epochs = 150
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         for batch in train_loader:
#             inputs, labels = batch
#             vertices = inputs["vertices"].to(device)
#             normals = inputs["normals"].to(device)
#             labels = labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(vertices, normals)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()

#         scheduler.step()
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

#     # Evaluation
#     model.eval()
#     y_true, y_pred = [], []
#     with torch.no_grad():
#         for batch in test_loader:
#             inputs, labels = batch
#             vertices = inputs["vertices"].to(device)
#             normals = inputs["normals"].to(device)
#             labels = labels.to(device)

#             outputs = model(vertices, normals)
#             _, preds = torch.max(outputs, 1)

#             y_true.extend(labels.cpu().numpy())
#             y_pred.extend(preds.cpu().numpy())

#     # Metrics
#     acc = accuracy_score(y_true, y_pred)
#     prec = precision_score(y_true, y_pred, average='weighted', zero_division=1)
#     rec = recall_score(y_true, y_pred, average='weighted', zero_division=1)
#     f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
#     conf_matrix = confusion_matrix(y_true, y_pred)

#     print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")
#     print("Confusion Matrix:")
#     print(conf_matrix)

#     # Plot Confusion Matrix
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
#     plt.title("Confusion Matrix")
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Confusion matrix values
conf_matrix = np.array([
    [1337, 29, 8, 7],
    [56, 36, 0, 1],
    [33, 2, 1, 1],
    [16, 0, 0, 0]
])

# Labels for the matrix
labels = ["Class 0", "Class 1", "Class 2", "Class 3"]

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
