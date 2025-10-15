"""
Stomata Counting Regression Model

This script trains a regression model to count stomata in leaf microscope images using classical ML techniques.

Workflow:
1. Load CSV with image filenames and manual counts
2. Preprocess images: resize to 512x512, convert to grayscale, normalize to 0-1
3. Extract features: histogram (256 bins), GLCM texture (contrast, energy, homogeneity), edge density
4. Train RandomForestRegressor and SVR models
5. Evaluate with MAE and visualize predictions

Requirements: pandas, numpy, pillow, opencv-python, scikit-image, scikit-learn, matplotlib

Run: python train.py
"""

import pandas as pd
import numpy as np
from PIL import Image
import cv2
from skimage import feature, filters
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib

# Paths
csv_path = 'stomata/.data/nb_stomate.csv'
image_dir = 'stomata/.data/stomata/'

# Load CSV
df = pd.read_csv(csv_path, header=None, names=['filename', 'count'])
print("CSV loaded:")
print(df.head())
print(f"Total images: {len(df)}")

# Explore one image
sample_img_path = image_dir + df['filename'].iloc[0]
img = Image.open(sample_img_path)
print(f"Image size: {img.size}, mode: {img.mode}")
img_array = np.array(img)
print(f"Array shape: {img_array.shape}, dtype: {img_array.dtype}")
print(f"Value range: {img_array.min()} - {img_array.max()}")

# Determine preprocessing: assume grayscale, resize to 4096x4096, normalize to 0-1
def preprocess_image(img_path, size=(4096, 4096)):
    img = Image.open(img_path)
    if img.mode != 'L':
        img = img.convert('L')  # Convert to grayscale
    img = img.resize(size, Image.Resampling.LANCZOS)
    img_array = np.array(img) / 255.0  # Normalize
    return img_array

# Test preprocessing
sample_preprocessed = preprocess_image(sample_img_path)
print(f"Preprocessed shape: {sample_preprocessed.shape}, range: {sample_preprocessed.min()} - {sample_preprocessed.max()}")

# Feature extraction
def extract_features(img_array):
    # Ensure values are in [0, 1]
    img_array = np.clip(img_array, 0, 1)
    features = []
    # Histogram features
    img_uint = (img_array * 255).astype(np.uint8)
    hist = np.bincount(img_uint.flatten(), minlength=256)
    features.extend(hist / hist.sum())  # Normalize histogram

    # Texture features using GLCM
    from skimage.feature import graycomatrix, graycoprops
    img_uint = (img_array * 255).astype(np.uint8)
    glcm = graycomatrix(img_uint, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    features.extend([contrast, energy, homogeneity])

    # Edge density
    edges = filters.sobel(img_array)
    edge_density = np.mean(edges)
    features.append(edge_density)

    return np.array(features)

# Test feature extraction
sample_features = extract_features(sample_preprocessed)
print(f"Feature vector length: {len(sample_features)}")

# Prepare dataset
features_list = []
counts = []
for idx, row in df.iterrows():
    img_path = image_dir + row['filename']
    try:
        img_array = preprocess_image(img_path)
        features = extract_features(img_array)
        features_list.append(features)
        counts.append(row['count'])
    except FileNotFoundError:
        print(f"Image not found: {img_path}")
        continue

X = np.array(features_list)
y = np.array(counts)
print(f"Dataset shape: X={X.shape}, y={y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)

# Evaluate
y_pred_rf = rf.predict(X_test)
y_pred_svr = svr.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mae_svr = mean_absolute_error(y_test, y_pred_svr)

print(f"Random Forest MAE: {mae_rf}")
print(f"SVR MAE: {mae_svr}")

# Table of predicted vs true for test set
results_df = pd.DataFrame({
    'True Count': y_test,
    'RF Predicted': y_pred_rf,
    'SVR Predicted': y_pred_svr
})
print("\nTest Set Predictions:")
print(results_df.head(10))  # Show first 10

# Confidence for RF (std of predictions from individual trees)
rf_preds = np.array([tree.predict(X_test) for tree in rf.estimators_])
y_pred_rf_std = rf_preds.std(axis=0)
print(f"Random Forest mean prediction std: {y_pred_rf_std.mean():.2f}")
print(f"Random Forest prediction std range: {y_pred_rf_std.min():.2f} - {y_pred_rf_std.max():.2f}")

# Plot predictions
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_rf)
plt.xlabel('True counts')
plt.ylabel('Predicted counts')
plt.title('Random Forest')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_svr)
plt.xlabel('True counts')
plt.ylabel('Predicted counts')
plt.title('SVR')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')

plt.tight_layout()
plt.savefig('stomata/predictions.png')
plt.show()

# Save the best model (RF)
joblib.dump(rf, 'stomata/.data/rf_model.pkl')
print("Model saved to stomata/.data/rf_model.pkl")

# Inference function
def predict_stomata(image_path, model):
    img_array = preprocess_image(image_path)
    features = extract_features(img_array)
    prediction = model.predict([features])[0]
    return prediction

# Predict on unclassified images
import os
unclassified_dir = 'stomata/.data/unclassified/'
if os.path.exists(unclassified_dir):
    unclassified_files = [f for f in os.listdir(unclassified_dir) if f.endswith('.tif')]
    print(f"\nPredicting on {len(unclassified_files)} unclassified images:")
    for filename in unclassified_files[:5]:  # Show first 5
        img_path = os.path.join(unclassified_dir, filename)
        try:
            prediction = predict_stomata(img_path, rf)
            print(f"{filename}: Predicted {prediction:.1f} stomata")
        except Exception as e:
            print(f"{filename}: Error - {e}")
else:
    print("Unclassified directory not found.")

# Example usage
# prediction = predict_stomata('path/to/new/image.tif')
# print(f"Predicted stomata count: {prediction}")