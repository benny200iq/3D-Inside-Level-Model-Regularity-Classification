import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import trimesh
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns



# Load labels from Excel file
label_file = 'Dataset/3d-future-dataset/label/Final_Validated_Regularity_Levels.xlsx'
label_data = pd.read_excel(label_file)

# Extract relevant columns
labels = label_data[['Object ID (Dataset Original Object ID)', 'Final Inside Level']]

# Set the maximum number of data points to use
MAX = len(label_data)
MAX_VALUE = MAX

# Limit the number of data points to use
labels = labels[:MAX_VALUE]

# Path to the folder containing 3D objects
obj_folder = 'Dataset/3d-future-dataset/obj-3d.future'

# Feature extraction function
def extract_features_from_obj(file_path):
    try:
        mesh = trimesh.load(file_path, force='mesh')
        # Enhanced feature extraction: number of vertices, number of faces, surface area, volume, edge length, and bounding box volume
        num_vertices = len(mesh.vertices)
        num_faces = len(mesh.faces)
        surface_area = mesh.area
        volume = mesh.volume if mesh.is_volume else 0  # Some meshes may not have a well-defined volume
        edge_length = np.sum(mesh.edges_unique_length)
        bounding_box_volume = np.prod(mesh.bounding_box.extents)
        
        # Additional features like curvature could be added here if needed
        return [num_vertices, num_faces, surface_area, volume, edge_length, bounding_box_volume]
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Prepare dataset
features = []
targets = []

for index, row in tqdm(labels.iterrows(), total=len(labels)):
    object_id = row['Object ID (Dataset Original Object ID)']
    regularity_level = row['Final Inside Level']

    # Construct the path using FolderName and Object ID
    obj_file = os.path.join(obj_folder, object_id.strip(), f"normalized_model.obj")
    
    # Extract features
    if os.path.isfile(obj_file):
        feature_vector = extract_features_from_obj(obj_file)
        if feature_vector is not None:
            features.append(feature_vector)
            targets.append(regularity_level)

# Convert to numpy arrays
if len(features) == 0:
    print("No features extracted. Please check the dataset and feature extraction process.")
    exit()

X = np.array(features)
y = np.array(targets) - 1  # Adjust labels to be zero-indexed

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize classifiers
svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
xgb_clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=42)

# Initialize LabelBinarizer for multi-class AUC calculation
lb = LabelBinarizer()
lb.fit(y_train)
y_test_binarized = lb.transform(y_test)

# Train and evaluate SVM
print("\n••••••••••••••••••••••••••••••••••••••\n")
svm_clf.fit(X_train, y_train)
svm_predictions = svm_clf.predict(X_test)
svm_probabilities = svm_clf.predict_proba(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions, average='weighted', zero_division=0)
svm_recall = recall_score(y_test, svm_predictions, average='weighted', zero_division=0)
svm_f1 = f1_score(y_test, svm_predictions, average='weighted', zero_division=0)
svm_auc = roc_auc_score(y_test_binarized, svm_probabilities, multi_class='ovr', average='weighted')
svm_log_loss = log_loss(y_test, svm_probabilities)
svm_cm = confusion_matrix(y_test, svm_predictions)
print(f"SVM Accuracy: {svm_accuracy:.2f}")
print(f"SVM Precision: {svm_precision:.2f}")
print(f"SVM Recall: {svm_recall:.2f}")
print(f"SVM F1 Score: {svm_f1:.2f}")
print(f"SVM AUC-ROC: {svm_auc:.2f}")
print(f"SVM Log Loss: {svm_log_loss:.2f}")
print("\nConfusion Matrix:")
print(svm_cm)

# Train and evaluate Random Forest
print("\n••••••••••••••••••••••••••••••••••••••\n")
rf_clf.fit(X_train, y_train)
rf_predictions = rf_clf.predict(X_test)
rf_probabilities = rf_clf.predict_proba(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions, average='weighted', zero_division=0)
rf_recall = recall_score(y_test, rf_predictions, average='weighted', zero_division=0)
rf_f1 = f1_score(y_test, rf_predictions, average='weighted', zero_division=0)
rf_auc = roc_auc_score(y_test_binarized, rf_probabilities, multi_class='ovr', average='weighted')
rf_log_loss = log_loss(y_test, rf_probabilities)
rf_cm = confusion_matrix(y_test, rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
print(f"Random Forest Precision: {rf_precision:.2f}")
print(f"Random Forest Recall: {rf_recall:.2f}")
print(f"Random Forest F1 Score: {rf_f1:.2f}")
print(f"Random Forest AUC-ROC: {rf_auc:.2f}")
print(f"Random Forest Log Loss: {rf_log_loss:.2f}")
print("\nConfusion Matrix:")
print(rf_cm)


# Train and evaluate XGBoost
print("\n••••••••••••••••••••••••••••••••••••••\n")
xgb_clf.fit(X_train, y_train)
xgb_predictions = xgb_clf.predict(X_test)
xgb_probabilities = xgb_clf.predict_proba(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
xgb_precision = precision_score(y_test, xgb_predictions, average='weighted', zero_division=0)
xgb_recall = recall_score(y_test, xgb_predictions, average='weighted', zero_division=0)
xgb_f1 = f1_score(y_test, xgb_predictions, average='weighted', zero_division=0)
xgb_auc = roc_auc_score(y_test_binarized, xgb_probabilities, multi_class='ovr', average='weighted')
xgb_log_loss = log_loss(y_test, xgb_probabilities)
xgb_cm = confusion_matrix(y_test, xgb_predictions)
print(f"XGBoost Accuracy: {xgb_accuracy:.2f}")
print(f"XGBoost Precision: {xgb_precision:.2f}")
print(f"XGBoost Recall: {xgb_recall:.2f}")
print(f"XGBoost F1 Score: {xgb_f1:.2f}")
print(f"XGBoost AUC-ROC: {xgb_auc:.2f}")
print(f"XGBoost Log Loss: {xgb_log_loss:.2f}")
print("\nConfusion Matrix:")
print(xgb_cm)


# Plot all confusion matrices at the same time
plt.figure(figsize=(24, 6))

plt.subplot(1, 3, 1)
sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues')
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 3, 2)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 3, 3)
sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Blues')
plt.title('XGBoost Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()

# Compare model accuracies
print("\n••••••••••••••••••••••••••••••••••••••")
print("\nModel Comparison:")
print(f"SVM Accuracy: {svm_accuracy:.2f}")
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
print(f"XGBoost Accuracy: {xgb_accuracy:.2f}")
