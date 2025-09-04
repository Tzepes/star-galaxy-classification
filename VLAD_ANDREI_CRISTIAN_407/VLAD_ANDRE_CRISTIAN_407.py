import kagglehub

# Download latest version
path = kagglehub.dataset_download("divyansh22/dummy-astronomy-data")

print("Path to dataset files:", path)
import os
from IPython.display import display
from PIL import Image

import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
# List all files in the dataset directory
files = os.listdir(path)
print("Files:", files)
# Define paths to galaxy and star image folders
galaxy_folder = os.path.join(path, "Cutout Files", "galaxy")
star_folder = os.path.join(path, "Cutout Files", "star")

print("Galaxy folder:", galaxy_folder)
print("List of galaxy images:", os.listdir(galaxy_folder))
print("Star folder:", star_folder)
print("List of star images:", os.listdir(star_folder))
# Get full paths to images and their names
galaxy_images_paths = [os.path.join(galaxy_folder, f) for f in os.listdir(galaxy_folder)]
star_images_paths = [os.path.join(star_folder, f) for f in os.listdir(star_folder)]

galaxy_images_names = [os.path.basename(f) for f in galaxy_images_paths]
star_images_names = [os.path.basename(f) for f in star_images_paths]

print("Image paths for galaxies:", galaxy_images_paths[:5])  # Print first 5 galaxy image paths
print("Image paths for stars:", star_images_paths[:5])      # Print first 5 star image paths

print("Number of galaxy images:", len(galaxy_images_paths))
print("Number of star images:", len(star_images_paths))

print(galaxy_images_names[:5])  # Print first 5 galaxy image paths
print(star_images_names[:5])    # Print first 5 star image paths
# Display first 3 galaxy images
print("Galaxy Images:")
for img_path in galaxy_images_paths[:3]:
    img = Image.open(img_path)
    display(img)

# Display first 3 star images
print("Star Images:")
for img_path in star_images_paths[:3]:
    img = Image.open(img_path)
    display(img)
# outpute image sizes
galaxy_sizes = [Image.open(img).size for img in galaxy_images_paths[:5]]
star_sizes = [Image.open(img).size for img in star_images_paths[:5]]

print("Galaxy image sizes (first 5):", galaxy_sizes)
print("Star image sizes (first 5):", star_sizes)
#include top = false to exclude the fully connected layers (we only need feature extraction)
vgg16_model = VGG16(weights='imagenet', include_top=False)

def preprocess_image(img_path): # Preprocess a single image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array
X = None
y = None
# Create a list to store features and labels

all_features = []
all_labels = []

# Function to extract features using VGG16
def extract_features(images_paths, label):
    features = []
    labels = []

    for img_path in images_paths:
        processed_img = preprocess_image(img_path)
        feature_vector = vgg16_model.predict(processed_img)
        feature_vector = feature_vector.flatten()
        features.append(feature_vector)
        labels.append(label)

    return features, labels
# Process star images
star_features, star_labels = extract_features(star_images_paths, 0)
all_features.extend(star_features)
all_labels.extend(star_labels)

# Process galaxy images
galaxy_features, galaxy_labels = extract_features(galaxy_images_paths, 1)
all_features.extend(galaxy_features)
all_labels.extend(galaxy_labels)

# Convert lists to numpy arrays with correct shape
X = np.array(all_features)
y = np.array(all_labels)

# Print the shape of the resulting feature matrix
print(f"Shape of feature matrix X: {X.shape}")
print(f"Shape of labels y: {y.shape}")

# Save the feature matrix X and labels y to files
features_file = 'features.npy'
labels_file = 'labels.npy'

np.save(features_file, X)
np.save(labels_file, y)

print(f"Feature matrix X saved to {features_file}")
print(f"Labels y saved to {labels_file}")
features_file = 'features.npy'
labels_file = 'labels.npy'

# Check if X and y are defined and not None, otherwise load from files
# This is due to the fact that processing images with VGG takes very long,
# so initial processings are saved to a file. If code has to be ran without the need
# of reprocessing, access the files.
if X is None or y is None:
    if os.path.exists(features_file) and os.path.exists(labels_file):
        print(f"Loading feature matrix X from {features_file}")
        X = np.load(features_file, allow_pickle=True)
        # Reshape the loaded features array
        num_samples = len(X) # Get number of samples from loaded data
        feature_dim = 25088  # This is the expected dimension of the VGG16 features
        X = X.reshape(num_samples, feature_dim)
        print(f"Loading labels y from {labels_file}")
        y = np.load(labels_file, allow_pickle=True)
        print(f"Shape of loaded feature matrix X: {X.shape}")
        print(f"Shape of loaded labels y: {y.shape}")
    else:
        print("Feature and label files not found. Please run the cells to generate them.")
        # Exit or handle the case where files are not found and X and y are None
else:
    print("X and y are already defined.")


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- KNN Model ---
print("\nTraining K-Nearest Neighbors Classifier...")
# We use a small value for n_neighbors as a starting point.
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)
print(f"KNN Accuracy: {knn_accuracy:.4f}")

# --- SVM Model ---
print("\nTraining Support Vector Machine Classifier...")
# SVM can be computationally expensive on large datasets, so we
# use a linear kernel for a quicker training time.
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f"SVM Accuracy: {svm_accuracy:.4f}")
features_file = 'features.npy'
labels_file = 'labels.npy'

# Load the arrays from the files
loaded_features = np.load(features_file, allow_pickle=True)
loaded_labels = np.load(labels_file, allow_pickle=True)

# Reshape the loaded features array
num_samples = len(loaded_labels)
feature_dim = 25088  # This is the expected dimension of the VGG16 features
loaded_features = loaded_features.reshape(num_samples, feature_dim)


# Print the shapes of the loaded arrays
print(f"Shape of loaded features: {loaded_features.shape}")
print(f"Shape of loaded labels: {loaded_labels.shape}")

# Display the first few elements of each array
print("\nFirst 5 elements of features:")
display(loaded_features[:5])

print("\nFirst 5 elements of labels:")
display(loaded_labels[:5])
features_file_py = 'features.npy'
labels_file_py = 'labels.npy'

if os.path.exists(f'{features_file_py}'):
    print(f"{features_file_py} exists.")
else:
    print(f"{features_file_py} does not exist.")

if os.path.exists(labels_file_py):
    print(f"{labels_file_py} exists.")
else:
    print(f"{labels_file_py} does not exist.")

# Hyperparameter tuning for KNN using GridSearchCV
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 12],
    'leaf_size': [30, 40, 50],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# Initialize the KNN model
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1) # Use all available cores
grid_search.fit(X_train, y_train)
best_params_knn = None
best_score_knn = None
# Access the best parameters and best score
best_params_knn = grid_search.best_params_
best_score_knn = grid_search.best_score_

# Print the results
print("Best parameters found by GridSearchCV:")
print(best_params_knn)
print(f"\nBest cross-validation accuracy: {best_score_knn:.4f}")

if best_params_knn == None: 
# These are the best outpued parameters from previous runs. 
# Save this here in case of wanting to skip grid search durring development and testing.
  best_params_knn = { 
    'n_neighbors': 9,
    'leaf_size': 30,
    'metric': 'manhattan'
  }
# Train a final KNN model with the best parameters
final_knn_model = KNeighborsClassifier(**best_params_knn)
final_knn_model.fit(X_train, y_train)
# Make predictions on the test set
final_knn_predictions = final_knn_model.predict(X_test)

# Calculate the accuracy
final_knn_accuracy = accuracy_score(y_test, final_knn_predictions)

# Print the accuracy
print(f"Final KNN Model Accuracy on Test Set: {final_knn_accuracy:.4f}")
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}
svm = SVC(random_state=42)
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=5, n_jobs=-1)
grid_search_svm.fit(X_train, y_train)
# Access the best parameters and best score for SVM
best_params_svm = grid_search_svm.best_params_
best_score_svm = grid_search_svm.best_score_

# Print the results for SVM
print("Best parameters found by GridSearchCV for SVM:")
print(best_params_svm)
print(f"\nBest cross-validation accuracy for SVM: {best_score_svm:.4f}")
# Train a final SVM model with the best parameters
final_svm_model = SVC(**best_params_svm, random_state=42)
final_svm_model.fit(X_train, y_train)
# Make predictions on the test set using the final SVM model
final_svm_predictions = final_svm_model.predict(X_test)

# Calculate the accuracy of the final SVM model
final_svm_accuracy = accuracy_score(y_test, final_svm_predictions)

# Print the accuracy
print(f"Final SVM Model Accuracy on Test Set: {final_svm_accuracy:.4f}")
# Confusion Matrix for KNN
knn_cm = confusion_matrix(y_test, final_knn_predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(knn_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Star', 'Galaxy'], yticklabels=['Star', 'Galaxy'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for KNN Model')
plt.show()

# Confusion Matrix for SVM
svm_cm = confusion_matrix(y_test, final_svm_predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Star', 'Galaxy'], yticklabels=['Star', 'Galaxy'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for SVM Model')
plt.show()