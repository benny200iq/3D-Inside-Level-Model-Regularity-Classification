import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import warnings
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import trimesh
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Load labels from Excel file
label_file = 'Dataset/pix3d/label/Final_Validated_Regularity_Levels.xlsx'
label_data = pd.read_excel(label_file)

# Extract relevant columns
labels = label_data[['Object ID (Dataset Original Object ID)', 'Final Inside Level']]

# Set the maximum number of data points to use
MAX = len(label_data)
MAX_VALUE = MAX

# Limit the number of data points to use
labels = labels[:MAX_VALUE]

# Path to the folder containing 3D objects
obj_folder = 'Dataset/pix3d/obj-pix3d'

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
    obj_file = os.path.join(obj_folder, 'models', object_id.strip(), f"model.obj")
    
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

# Check the distribution of classes
print("Class distribution:")
print(Counter(y))

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize classifiers
svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
xgb_clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=42)

# Initialize LabelBinarizer for multi-class AUC calculation
lb = LabelBinarizer()
lb.fit(y_train)

warnings.filterwarnings('ignore', category=UserWarning, message='The least populated class in y has only.*')

# Cross-validation setup
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Initialize confusion matrices
svm_cm = np.zeros((len(lb.classes_), len(lb.classes_)))
rf_cm = np.zeros((len(lb.classes_), len(lb.classes_)))
xgb_cm = np.zeros((len(lb.classes_), len(lb.classes_)))

# Train and evaluate SVM with cross-validation
print("\nSVM Cross-Validation\n" + "-"*30)
svm_auc_scores = []
svm_accuracy_scores = []
svm_precision_scores = []
svm_recall_scores = []
svm_f1_scores = []
svm_log_losses = []
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    svm_clf.fit(X_train, y_train)
    svm_predictions = svm_clf.predict(X_test)
    svm_probabilities = svm_clf.predict_proba(X_test)

    y_test_binarized = lb.transform(y_test)

    # Update confusion matrix
    svm_cm += confusion_matrix(y_test, svm_predictions, labels=lb.classes_)

    # Calculate metrics
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    svm_precision = precision_score(y_test, svm_predictions, average='weighted', zero_division=0)
    svm_recall = recall_score(y_test, svm_predictions, average='weighted', zero_division=0)
    svm_f1 = f1_score(y_test, svm_predictions, average='weighted', zero_division=0)
    svm_log_loss_value = log_loss(y_test, svm_probabilities, labels=lb.classes_)

    svm_accuracy_scores.append(svm_accuracy)
    svm_precision_scores.append(svm_precision)
    svm_recall_scores.append(svm_recall)
    svm_f1_scores.append(svm_f1)
    svm_log_losses.append(svm_log_loss_value)

    # Calculate AUC-ROC if applicable
    if y_test_binarized.shape[1] > 1 and len(set(y_test)) > 1:
        try:
            svm_auc = roc_auc_score(y_test_binarized, svm_probabilities, multi_class='ovr', average='weighted')
            svm_auc_scores.append(svm_auc)
        except ValueError as e:
            print(f"AUC-ROC calculation skipped: {e}")

if len(svm_auc_scores) > 0:
    print(f"Average SVM AUC-ROC: {np.mean(svm_auc_scores):.2f}")
else:
    print("No valid AUC-ROC scores were calculated.")

print(f"Average SVM Accuracy: {np.mean(svm_accuracy_scores):.2f}")
print(f"Average SVM Precision: {np.mean(svm_precision_scores):.2f}")
print(f"Average SVM Recall: {np.mean(svm_recall_scores):.2f}")
print(f"Average SVM F1 Score: {np.mean(svm_f1_scores):.2f}")
print(f"Average SVM Log Loss: {np.mean(svm_log_losses):.2f}")

# Train and evaluate Random Forest with cross-validation
print("\nRandom Forest Cross-Validation\n" + "-"*30)
rf_auc_scores = []
rf_accuracy_scores = []
rf_precision_scores = []
rf_recall_scores = []
rf_f1_scores = []
rf_log_losses = []
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    rf_clf.fit(X_train, y_train)
    rf_predictions = rf_clf.predict(X_test)
    rf_probabilities = rf_clf.predict_proba(X_test)

    y_test_binarized = lb.transform(y_test)

    # Update confusion matrix
    rf_cm += confusion_matrix(y_test, rf_predictions, labels=lb.classes_)

    # Calculate metrics
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    rf_precision = precision_score(y_test, rf_predictions, average='weighted', zero_division=0)
    rf_recall = recall_score(y_test, rf_predictions, average='weighted', zero_division=0)
    rf_f1 = f1_score(y_test, rf_predictions, average='weighted', zero_division=0)
    rf_log_loss_value = log_loss(y_test, rf_probabilities, labels=lb.classes_)

    rf_accuracy_scores.append(rf_accuracy)
    rf_precision_scores.append(rf_precision)
    rf_recall_scores.append(rf_recall)
    rf_f1_scores.append(rf_f1)
    rf_log_losses.append(rf_log_loss_value)

    # Calculate AUC-ROC if applicable
    if y_test_binarized.shape[1] > 1 and len(set(y_test)) > 1:
        try:
            rf_auc = roc_auc_score(y_test_binarized, rf_probabilities, multi_class='ovr', average='weighted')
            rf_auc_scores.append(rf_auc)
        except ValueError as e:
            print(f"AUC-ROC calculation skipped: {e}")

if len(rf_auc_scores) > 0:
    print(f"Average Random Forest AUC-ROC: {np.mean(rf_auc_scores):.2f}")
else:
    print("No valid AUC-ROC scores were calculated.")

print(f"Average Random Forest Accuracy: {np.mean(rf_accuracy_scores):.2f}")
print(f"Average Random Forest Precision: {np.mean(rf_precision_scores):.2f}")
print(f"Average Random Forest Recall: {np.mean(rf_recall_scores):.2f}")
print(f"Average Random Forest F1 Score: {np.mean(rf_f1_scores):.2f}")
print(f"Average Random Forest Log Loss: {np.mean(rf_log_losses):.2f}")

# Train and evaluate XGBoost with cross-validation
print("\nXGBoost Cross-Validation\n" + "-"*30)
xgb_auc_scores = []
xgb_accuracy_scores = []
xgb_precision_scores = []
xgb_recall_scores = []
xgb_f1_scores = []
xgb_log_losses = []
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    xgb_clf.fit(X_train, y_train)
    xgb_predictions = xgb_clf.predict(X_test)
    xgb_probabilities = xgb_clf.predict_proba(X_test)

    y_test_binarized = lb.transform(y_test)

    # Update confusion matrix
    xgb_cm += confusion_matrix(y_test, xgb_predictions, labels=lb.classes_)

    # Calculate metrics
    xgb_accuracy = accuracy_score(y_test, xgb_predictions)
    xgb_precision = precision_score(y_test, xgb_predictions, average='weighted', zero_division=0)
    xgb_recall = recall_score(y_test, xgb_predictions, average='weighted', zero_division=0)
    xgb_f1 = f1_score(y_test, xgb_predictions, average='weighted', zero_division=0)
    xgb_log_loss_value = log_loss(y_test, xgb_probabilities, labels=lb.classes_)

    xgb_accuracy_scores.append(xgb_accuracy)
    xgb_precision_scores.append(xgb_precision)
    xgb_recall_scores.append(xgb_recall)
    xgb_f1_scores.append(xgb_f1)
    xgb_log_losses.append(xgb_log_loss_value)

    # Calculate AUC-ROC if applicable
    if y_test_binarized.shape[1] > 1 and len(set(y_test)) > 1:
        try:
            xgb_auc = roc_auc_score(y_test_binarized, xgb_probabilities, multi_class='ovr', average='weighted')
            xgb_auc_scores.append(xgb_auc)
        except ValueError as e:
            print(f"AUC-ROC calculation skipped: {e}")

if len(xgb_auc_scores) > 0:
    print(f"Average XGBoost AUC-ROC: {np.mean(xgb_auc_scores):.2f}")
else:
    print("No valid AUC-ROC scores were calculated.")

print(f"Average XGBoost Accuracy: {np.mean(xgb_accuracy_scores):.2f}")
print(f"Average XGBoost Precision: {np.mean(xgb_precision_scores):.2f}")
print(f"Average XGBoost Recall: {np.mean(xgb_recall_scores):.2f}")
print(f"Average XGBoost F1 Score: {np.mean(xgb_f1_scores):.2f}")
print(f"Average XGBoost Log Loss: {np.mean(xgb_log_losses):.2f}")

# Plot confusion matrices
plt.figure(figsize=(24, 6))

plt.subplot(1, 3, 1)
sns.heatmap(svm_cm, annot=True, fmt='.0f', cmap='Blues')
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 3, 2)
sns.heatmap(rf_cm, annot=True, fmt='.0f', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 3, 3)
sns.heatmap(xgb_cm, annot=True, fmt='.0f', cmap='Blues')
plt.title('XGBoost Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()