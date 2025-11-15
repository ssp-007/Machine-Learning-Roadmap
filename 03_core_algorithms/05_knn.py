"""
K-Nearest Neighbors (KNN) - Instance-based Learning
This tutorial covers:
1. KNN Classifier
2. KNN Regressor
3. Choosing the right K value
4. Distance metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns

print("=" * 60)
print("K-NEAREST NEIGHBORS (KNN) TUTORIAL")
print("=" * 60)

# ============================================
# Part 1: KNN Classifier
# ============================================
print("\n=== Part 1: KNN Classifier ===")
print("Classify based on K nearest neighbors")

# Create sample data: Iris-like classification
np.random.seed(42)
n_samples = 200

# Two features
feature1 = np.random.randn(n_samples) * 2
feature2 = np.random.randn(n_samples) * 2

# Create two classes based on distance from origin
distance = np.sqrt(feature1**2 + feature2**2)
class_labels = (distance > 2).astype(int)

# Add some noise
class_labels = class_labels ^ (np.random.rand(n_samples) < 0.1).astype(int)

# Create DataFrame
data = pd.DataFrame({
    'feature1': feature1,
    'feature2': feature2,
    'class': class_labels
})

# Prepare data
X = data[['feature1', 'feature2']]
y = data['class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (important for KNN!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = knn_model.predict(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Performance (K=5):")
print(f"  Accuracy: {accuracy:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(cm)

# Visualize decision boundary
h = 0.02
x_min, x_max = X['feature1'].min() - 1, X['feature1'].max() + 1
y_min, y_max = X['feature2'].min() - 1, X['feature2'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = knn_model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
scatter = plt.scatter(X_test['feature1'], X_test['feature2'], 
                     c=y_test, cmap='RdYlBu', edgecolors='black', s=100)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KNN Decision Boundary (K=5)')
plt.colorbar(scatter)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('knn_decision_boundary.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved plot: knn_decision_boundary.png")

# ============================================
# Part 2: Choosing the Right K Value
# ============================================
print("\n=== Part 2: Choosing the Right K Value ===")
print("K too small = overfitting, K too large = underfitting")

k_values = range(1, 31)
train_scores = []
test_scores = []

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_scaled, y_train)
    
    train_score = knn_temp.score(X_train_scaled, y_train)
    test_score = knn_temp.score(X_test_scaled, y_test)
    
    train_scores.append(train_score)
    test_scores.append(test_score)

# Find best K
best_k_idx = np.argmax(test_scores)
best_k = k_values[best_k_idx]

print(f"\nBest K value: {best_k} (Test Accuracy: {test_scores[best_k_idx]:.4f})")

# Visualize
plt.figure(figsize=(12, 6))
plt.plot(k_values, train_scores, 'o-', label='Training Accuracy', linewidth=2)
plt.plot(k_values, test_scores, 's-', label='Test Accuracy', linewidth=2)
plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best K={best_k}')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('KNN: Effect of K Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('knn_k_selection.png', dpi=150, bbox_inches='tight')
print("✅ Saved plot: knn_k_selection.png")
print(f"\nObservations:")
print(f"  - K=1: Perfect training accuracy (memorizes data)")
print(f"  - K too large: Underfitting (ignores local patterns)")
print(f"  - Best K: Balance between bias and variance")

# ============================================
# Part 3: KNN Regressor
# ============================================
print("\n=== Part 3: KNN Regressor ===")
print("Predicting continuous values using K nearest neighbors")

# Create regression data
np.random.seed(42)
n_samples = 150

# Feature
X_reg = np.random.rand(n_samples, 1) * 10

# Target: Non-linear relationship
y_reg = np.sin(X_reg.flatten()) * 3 + X_reg.flatten() * 0.5 + np.random.randn(n_samples) * 0.5

# Split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Scale
scaler_reg = StandardScaler()
X_train_r_scaled = scaler_reg.fit_transform(X_train_r)
X_test_r_scaled = scaler_reg.transform(X_test_r)

# Train model
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train_r_scaled, y_train_r)

# Predict
y_pred_r = knn_reg.predict(X_test_r_scaled)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test_r, y_pred_r)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_r, y_pred_r)

print(f"\nModel Performance (K=5):")
print(f"  R² Score: {r2:.4f}")
print(f"  RMSE: {rmse:.4f}")

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
X_plot = np.linspace(X_reg.min(), X_reg.max(), 100).reshape(-1, 1)
X_plot_scaled = scaler_reg.transform(X_plot)
y_plot = knn_reg.predict(X_plot_scaled)

plt.scatter(X_train_r, y_train_r, alpha=0.6, label='Training Data')
plt.plot(X_plot, y_plot, 'r-', label='KNN Prediction', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('KNN Regression (K=5)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y_test_r, y_pred_r, alpha=0.6)
plt.plot([y_test_r.min(), y_test_r.max()], 
         [y_test_r.min(), y_test_r.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Predicted vs Actual')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('knn_regression.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved plot: knn_regression.png")

# ============================================
# Part 4: Distance Metrics
# ============================================
print("\n=== Part 4: Distance Metrics ===")
print("Different ways to measure 'nearest'")

from sklearn.metrics import pairwise_distances

# Sample points
points = np.array([[0, 0], [3, 4], [1, 1]])

# Euclidean distance (default)
euclidean = pairwise_distances(points, metric='euclidean')
print(f"\nEuclidean Distance:")
print(euclidean)

# Manhattan distance
manhattan = pairwise_distances(points, metric='manhattan')
print(f"\nManhattan Distance:")
print(manhattan)

# Compare KNN with different metrics
metrics = ['euclidean', 'manhattan', 'minkowski']
metric_scores = []

for metric in metrics:
    knn_metric = KNeighborsClassifier(n_neighbors=5, metric=metric)
    knn_metric.fit(X_train_scaled, y_train)
    score = knn_metric.score(X_test_scaled, y_test)
    metric_scores.append(score)
    print(f"  {metric.capitalize()}: {score:.4f}")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\n✅ KNN: Classify/predict based on K nearest neighbors")
print("✅ Instance-based learning (lazy learning)")
print("✅ No training phase (just stores data)")
print("\nKey Parameters:")
print("  - n_neighbors (K): Number of neighbors to consider")
print("  - metric: Distance metric (euclidean, manhattan, etc.)")
print("  - weights: Uniform or distance-weighted")
print("\nPros:")
print("  - Simple and intuitive")
print("  - No assumptions about data distribution")
print("  - Works well for non-linear problems")
print("  - Can be used for both classification and regression")
print("\nCons:")
print("  - Slow for large datasets (computes distances)")
print("  - Sensitive to irrelevant features")
print("  - Requires feature scaling")
print("  - Sensitive to K value")
print("\nNext: Try svm.py (Support Vector Machines)!")

