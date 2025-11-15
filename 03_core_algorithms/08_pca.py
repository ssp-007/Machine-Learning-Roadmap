"""
Principal Component Analysis (PCA) - Dimensionality Reduction
This tutorial covers:
1. PCA basics
2. Explained variance
3. Dimensionality reduction
4. Visualization of high-dimensional data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import seaborn as sns

print("=" * 60)
print("PRINCIPAL COMPONENT ANALYSIS (PCA) TUTORIAL")
print("=" * 60)

# ============================================
# Part 1: PCA Basics
# ============================================
print("\n=== Part 1: PCA Basics ===")
print("Reducing dimensions while keeping most information")

# Create sample 2D data
np.random.seed(42)
n_samples = 100

# Create correlated data
X_2d = np.random.randn(n_samples, 2)
X_2d[:, 1] = X_2d[:, 0] * 0.8 + np.random.randn(n_samples) * 0.3

# Apply PCA
pca_2d = PCA(n_components=2)
X_2d_pca = pca_2d.fit_transform(X_2d)

print(f"\nOriginal Data Shape: {X_2d.shape}")
print(f"PCA Transformed Shape: {X_2d_pca.shape}")
print(f"\nExplained Variance Ratio:")
for i, ratio in enumerate(pca_2d.explained_variance_ratio_):
    print(f"  PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Original data
axes[0].scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.6)
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].set_title('Original Data')
axes[0].grid(True, alpha=0.3)

# Add principal components
mean = np.mean(X_2d, axis=0)
for i, (length, vector) in enumerate(zip(pca_2d.explained_variance_, pca_2d.components_)):
    v = vector * np.sqrt(length) * 2
    axes[0].arrow(mean[0], mean[1], v[0], v[1], 
                 head_width=0.2, head_length=0.2, fc='red', ec='red', linewidth=2)
    axes[0].text(mean[0] + v[0]*1.2, mean[1] + v[1]*1.2, f'PC{i+1}', 
                fontsize=12, color='red', weight='bold')

# PCA transformed data
axes[1].scatter(X_2d_pca[:, 0], X_2d_pca[:, 1], alpha=0.6)
axes[1].set_xlabel('Principal Component 1')
axes[1].set_ylabel('Principal Component 2')
axes[1].set_title('PCA Transformed Data')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1].axvline(x=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('pca_basics.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved plot: pca_basics.png")

# ============================================
# Part 2: Dimensionality Reduction
# ============================================
print("\n=== Part 2: Dimensionality Reduction ===")
print("Reducing from high dimensions to lower dimensions")

# Create high-dimensional data
np.random.seed(42)
n_samples = 200
n_features = 10

# Create data with some correlation
X_high = np.random.randn(n_samples, n_features)
# Make some features correlated
X_high[:, 2] = X_high[:, 0] * 0.7 + X_high[:, 1] * 0.3 + np.random.randn(n_samples) * 0.2
X_high[:, 5] = X_high[:, 3] * 0.6 + np.random.randn(n_samples) * 0.3

# Scale data (important for PCA!)
scaler = StandardScaler()
X_high_scaled = scaler.fit_transform(X_high)

# Apply PCA with different numbers of components
n_components_list = [1, 2, 3, 5, 10]
explained_variances = []

for n_comp in n_components_list:
    pca_temp = PCA(n_components=n_comp)
    pca_temp.fit(X_high_scaled)
    explained_var = np.sum(pca_temp.explained_variance_ratio_)
    explained_variances.append(explained_var)
    print(f"  {n_comp} component(s): {explained_var:.4f} ({explained_var*100:.2f}% variance explained)")

# Visualize explained variance
pca_full = PCA()
pca_full.fit(X_high_scaled)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
         np.cumsum(pca_full.explained_variance_ratio_), 'o-', linewidth=2)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Number of Components')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
        pca_full.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Individual Component Variance')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('pca_explained_variance.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved plot: pca_explained_variance.png")

# Reduce to 2D
pca_2 = PCA(n_components=2)
X_reduced = pca_2.fit_transform(X_high_scaled)

print(f"\nDimensionality Reduction:")
print(f"  Original dimensions: {X_high_scaled.shape[1]}")
print(f"  Reduced dimensions: {X_reduced.shape[1]}")
print(f"  Variance retained: {np.sum(pca_2.explained_variance_ratio_):.4f}")

# ============================================
# Part 3: PCA on Iris Dataset
# ============================================
print("\n=== Part 3: PCA on Real Dataset (Iris) ===")

# Load Iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target
feature_names = iris.feature_names

print(f"\nIris Dataset:")
print(f"  Samples: {X_iris.shape[0]}")
print(f"  Features: {X_iris.shape[1]}")
print(f"  Feature names: {feature_names}")

# Scale
scaler_iris = StandardScaler()
X_iris_scaled = scaler_iris.fit_transform(X_iris)

# Apply PCA
pca_iris = PCA(n_components=2)
X_iris_pca = pca_iris.fit_transform(X_iris_scaled)

print(f"\nPCA Results:")
print(f"  Explained variance (PC1): {pca_iris.explained_variance_ratio_[0]:.4f}")
print(f"  Explained variance (PC2): {pca_iris.explained_variance_ratio_[1]:.4f}")
print(f"  Total variance explained: {np.sum(pca_iris.explained_variance_ratio_):.4f}")

# Visualize
plt.figure(figsize=(14, 6))

# Original data (first 2 features)
plt.subplot(1, 2, 1)
scatter1 = plt.scatter(X_iris[:, 0], X_iris[:, 1], c=y_iris, cmap='viridis', s=100, alpha=0.6)
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('Original Data (First 2 Features)')
plt.colorbar(scatter1, label='Class')
plt.grid(True, alpha=0.3)

# PCA transformed data
plt.subplot(1, 2, 2)
scatter2 = plt.scatter(X_iris_pca[:, 0], X_iris_pca[:, 1], c=y_iris, cmap='viridis', s=100, alpha=0.6)
plt.xlabel(f'PC1 ({pca_iris.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca_iris.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title('PCA Transformed Data (2D)')
plt.colorbar(scatter2, label='Class')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_iris.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved plot: pca_iris.png")
print("→ Notice how classes are better separated in PCA space!")

# ============================================
# Part 4: PCA for Feature Reduction
# ============================================
print("\n=== Part 4: PCA for Feature Reduction ===")
print("Using PCA to reduce features before ML")

# Create classification problem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Use Iris data
X_train, X_test, y_train, y_test = train_test_split(
    X_iris_scaled, y_iris, test_size=0.2, random_state=42
)

# Train without PCA
rf_full = RandomForestClassifier(random_state=42)
rf_full.fit(X_train, y_train)
y_pred_full = rf_full.predict(X_test)
acc_full = accuracy_score(y_test, y_pred_full)

# Train with PCA (2 components)
pca_ml = PCA(n_components=2)
X_train_pca = pca_ml.fit_transform(X_train)
X_test_pca = pca_ml.transform(X_test)

rf_pca = RandomForestClassifier(random_state=42)
rf_pca.fit(X_train_pca, y_train)
y_pred_pca = rf_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)

print(f"\nModel Performance:")
print(f"  Without PCA (4 features): {acc_full:.4f}")
print(f"  With PCA (2 components):  {acc_pca:.4f}")
print(f"  Accuracy difference:     {acc_full - acc_pca:.4f}")

if acc_pca >= acc_full * 0.95:
    print("\n→ PCA reduced features by 50% with minimal accuracy loss!")
else:
    print("\n→ Some information lost, but still reasonable performance")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\n✅ PCA: Reduces dimensions while keeping most information")
print("✅ Finds directions of maximum variance")
print("✅ Principal components are orthogonal (uncorrelated)")
print("\nKey Concepts:")
print("  - Explained Variance: How much information each PC captures")
print("  - Dimensionality Reduction: Fewer features, faster computation")
print("  - Visualization: Plot high-dimensional data in 2D/3D")
print("\nWhen to Use:")
print("  - High-dimensional data (many features)")
print("  - Features are correlated")
print("  - Need to visualize high-dimensional data")
print("  - Want to reduce computation time")
print("\nPros:")
print("  - Reduces overfitting")
print("  - Speeds up training")
print("  - Removes noise")
print("  - Helps visualization")
print("\nCons:")
print("  - Loses some information")
print("  - Hard to interpret principal components")
print("  - Requires feature scaling")
print("  - Linear transformation only")
print("\nNext: Try model_evaluation.py for comprehensive evaluation!")

