"""
Support Vector Machines (SVM) - Maximum Margin Classifier
This tutorial covers:
1. SVM Classifier (linear and non-linear)
2. Different kernels
3. C parameter (regularization)
4. Support vectors
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

print("=" * 60)
print("SUPPORT VECTOR MACHINES (SVM) TUTORIAL")
print("=" * 60)

# ============================================
# Part 1: Linear SVM
# ============================================
print("\n=== Part 1: Linear SVM ===")
print("Finds the best separating line (maximum margin)")

# Create linearly separable data
np.random.seed(42)
n_samples = 100

# Class 0
X0 = np.random.randn(n_samples//2, 2) + [2, 2]
y0 = np.zeros(n_samples//2)

# Class 1
X1 = np.random.randn(n_samples//2, 2) + [-2, -2]
y1 = np.ones(n_samples//2)

# Combine
X = np.vstack([X0, X1])
y = np.hstack([y0, y1])

# Create DataFrame
data = pd.DataFrame(X, columns=['feature1', 'feature2'])
data['class'] = y

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (important for SVM!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train linear SVM
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svm_linear.predict(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\nLinear SVM Performance:")
print(f"  Accuracy: {accuracy:.4f}")

# Support Vectors
n_support = len(svm_linear.support_vectors_)
print(f"  Number of Support Vectors: {n_support}")

# Visualize
def plot_svm_decision_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', 
                         edgecolors='black', s=100)
    
    # Highlight support vectors
    support_vectors = model.support_vectors_
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
               s=200, facecolors='none', edgecolors='red', linewidths=2,
               label='Support Vectors')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar(scatter)
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.figure(figsize=(10, 8))
plot_svm_decision_boundary(svm_linear, X_test_scaled, y_test, 
                           'Linear SVM Decision Boundary')
plt.tight_layout()
plt.savefig('svm_linear.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved plot: svm_linear.png")

# ============================================
# Part 2: Non-linear SVM with RBF Kernel
# ============================================
print("\n=== Part 2: Non-linear SVM (RBF Kernel) ===")
print("Handles non-linearly separable data")

# Create non-linearly separable data (circles)
np.random.seed(42)
n_samples = 200

# Inner circle (class 0)
theta = np.random.uniform(0, 2*np.pi, n_samples//2)
r = np.random.uniform(0, 2, n_samples//2)
X0_nl = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
y0_nl = np.zeros(n_samples//2)

# Outer circle (class 1)
theta = np.random.uniform(0, 2*np.pi, n_samples//2)
r = np.random.uniform(3, 5, n_samples//2)
X1_nl = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
y1_nl = np.ones(n_samples//2)

# Combine
X_nl = np.vstack([X0_nl, X1_nl])
y_nl = np.hstack([y0_nl, y1_nl])

# Split
X_train_nl, X_test_nl, y_train_nl, y_test_nl = train_test_split(
    X_nl, y_nl, test_size=0.2, random_state=42, stratify=y_nl
)

# Scale
scaler_nl = StandardScaler()
X_train_nl_scaled = scaler_nl.fit_transform(X_train_nl)
X_test_nl_scaled = scaler_nl.transform(X_test_nl)

# Train with RBF kernel
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf.fit(X_train_nl_scaled, y_train_nl)

y_pred_nl = svm_rbf.predict(X_test_nl_scaled)
accuracy_nl = accuracy_score(y_test_nl, y_pred_nl)

print(f"\nRBF Kernel SVM Performance:")
print(f"  Accuracy: {accuracy_nl:.4f}")

# Compare with linear kernel
svm_linear_nl = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear_nl.fit(X_train_nl_scaled, y_train_nl)
y_pred_linear_nl = svm_linear_nl.predict(X_test_nl_scaled)
accuracy_linear_nl = accuracy_score(y_test_nl, y_pred_linear_nl)

print(f"\nLinear Kernel (for comparison):")
print(f"  Accuracy: {accuracy_linear_nl:.4f}")
print(f"  → RBF kernel is better for non-linear data!")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Linear kernel
h = 0.02
x_min, x_max = X_nl[:, 0].min() - 1, X_nl[:, 0].max() + 1
y_min, y_max = X_nl[:, 1].min() - 1, X_nl[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z_linear = svm_linear_nl.predict(scaler_nl.transform(np.c_[xx.ravel(), yy.ravel()]))
Z_linear = Z_linear.reshape(xx.shape)
axes[0].contourf(xx, yy, Z_linear, alpha=0.3, cmap='RdYlBu')
axes[0].scatter(X_test_nl[:, 0], X_test_nl[:, 1], c=y_test_nl, 
               cmap='RdYlBu', edgecolors='black', s=100)
axes[0].set_title(f'Linear Kernel (Accuracy: {accuracy_linear_nl:.3f})')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].grid(True, alpha=0.3)

# RBF kernel
Z_rbf = svm_rbf.predict(scaler_nl.transform(np.c_[xx.ravel(), yy.ravel()]))
Z_rbf = Z_rbf.reshape(xx.shape)
axes[1].contourf(xx, yy, Z_rbf, alpha=0.3, cmap='RdYlBu')
axes[1].scatter(X_test_nl[:, 0], X_test_nl[:, 1], c=y_test_nl, 
               cmap='RdYlBu', edgecolors='black', s=100)
axes[1].set_title(f'RBF Kernel (Accuracy: {accuracy_nl:.3f})')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('svm_kernels_comparison.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved plot: svm_kernels_comparison.png")

# ============================================
# Part 3: Effect of C Parameter
# ============================================
print("\n=== Part 3: Effect of C Parameter ===")
print("C controls the trade-off between margin and misclassification")

C_values = [0.1, 1, 10, 100, 1000]
train_scores = []
test_scores = []

for C in C_values:
    svm_temp = SVC(kernel='rbf', C=C, gamma='scale', random_state=42)
    svm_temp.fit(X_train_nl_scaled, y_train_nl)
    
    train_score = svm_temp.score(X_train_nl_scaled, y_train_nl)
    test_score = svm_temp.score(X_test_nl_scaled, y_test_nl)
    
    train_scores.append(train_score)
    test_scores.append(test_score)
    
    print(f"C={C:5.1f}: Train={train_score:.4f}, Test={test_score:.4f}")

# Visualize
plt.figure(figsize=(10, 6))
plt.plot(C_values, train_scores, 'o-', label='Training Accuracy', linewidth=2)
plt.plot(C_values, test_scores, 's-', label='Test Accuracy', linewidth=2)
plt.xscale('log')
plt.xlabel('C Parameter (log scale)')
plt.ylabel('Accuracy')
plt.title('SVM: Effect of C Parameter')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('svm_c_parameter.png', dpi=150, bbox_inches='tight')
print("✅ Saved plot: svm_c_parameter.png")
print("\nObservations:")
print("  - Low C: Larger margin, more misclassifications allowed")
print("  - High C: Smaller margin, fewer misclassifications (can overfit)")

# ============================================
# Part 4: Different Kernels
# ============================================
print("\n=== Part 4: Different Kernels ===")

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
kernel_scores = []

for kernel in kernels:
    svm_kernel = SVC(kernel=kernel, C=1.0, random_state=42)
    svm_kernel.fit(X_train_nl_scaled, y_train_nl)
    score = svm_kernel.score(X_test_nl_scaled, y_test_nl)
    kernel_scores.append(score)
    print(f"  {kernel.capitalize()} kernel: {score:.4f}")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\n✅ SVM: Finds the best separating hyperplane")
print("✅ Maximizes margin between classes")
print("✅ Support vectors define the decision boundary")
print("\nKey Parameters:")
print("  - C: Regularization (higher = less tolerance for misclassification)")
print("  - kernel: linear, poly, rbf, sigmoid")
print("  - gamma: Controls influence of each training example (RBF)")
print("\nPros:")
print("  - Effective in high-dimensional spaces")
print("  - Memory efficient (uses support vectors only)")
print("  - Versatile (different kernels)")
print("  - Works well with clear margin of separation")
print("\nCons:")
print("  - Doesn't perform well with large datasets")
print("  - Sensitive to feature scaling")
print("  - Not good with overlapping classes")
print("  - Black box (hard to interpret)")
print("\nNext: Try kmeans_clustering.py (Unsupervised Learning)!")

