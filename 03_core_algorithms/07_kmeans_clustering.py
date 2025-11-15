"""
K-Means Clustering - Unsupervised Learning
This tutorial covers:
1. K-Means clustering basics
2. Choosing the right K (Elbow method)
3. Clustering visualization
4. Real-world application: Customer segmentation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns

print("=" * 60)
print("K-MEANS CLUSTERING TUTORIAL")
print("=" * 60)

# ============================================
# Part 1: Basic K-Means Clustering
# ============================================
print("\n=== Part 1: Basic K-Means Clustering ===")
print("Grouping similar data points together")

# Create sample data with clear clusters
np.random.seed(42)

# Cluster 1
cluster1 = np.random.randn(50, 2) + [2, 2]

# Cluster 2
cluster2 = np.random.randn(50, 2) + [-2, -2]

# Cluster 3
cluster3 = np.random.randn(50, 2) + [2, -2]

# Combine
X = np.vstack([cluster1, cluster2, cluster3])
true_labels = np.hstack([np.zeros(50), np.ones(50), np.full(50, 2)])

# Create DataFrame
data = pd.DataFrame(X, columns=['feature1', 'feature2'])
data['true_cluster'] = true_labels

print(f"Data shape: {X.shape}")
print(f"Number of true clusters: {len(np.unique(true_labels))}")

# Visualize original data
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='viridis', s=100, alpha=0.6)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Original Data (True Clusters)')
plt.grid(True, alpha=0.3)

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)
predicted_labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print(f"\nK-Means Results:")
print(f"  Number of clusters: {len(np.unique(predicted_labels))}")
print(f"  Centroids:")
for i, centroid in enumerate(centroids):
    print(f"    Cluster {i}: ({centroid[0]:.2f}, {centroid[1]:.2f})")

# Visualize clusters
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis', s=100, alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', 
           s=200, label='Centroids', edgecolors='black', linewidths=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering (K=3)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kmeans_basic.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved plot: kmeans_basic.png")

# ============================================
# Part 2: Choosing the Right K (Elbow Method)
# ============================================
print("\n=== Part 2: Choosing the Right K (Elbow Method) ===")
print("Find optimal number of clusters")

# Try different K values
k_range = range(1, 11)
inertias = []
silhouette_scores = []

for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X)
    inertias.append(kmeans_temp.inertia_)
    
    if k > 1:  # Silhouette score needs at least 2 clusters
        score = silhouette_score(X, kmeans_temp.labels_)
        silhouette_scores.append(score)
    else:
        silhouette_scores.append(0)

# Find elbow (best K)
# Calculate rate of change
inertia_changes = np.diff(inertias)
inertia_changes_2 = np.diff(inertia_changes)
elbow_k = np.argmax(inertia_changes_2) + 2  # +2 because of double diff

print(f"\nElbow Method Results:")
print(f"  Suggested K (elbow): {elbow_k}")
print(f"  Inertia values:")
for k, inertia in zip(k_range, inertias):
    print(f"    K={k}: {inertia:.2f}")

# Best K based on silhouette score
best_k_silhouette = k_range[np.argmax(silhouette_scores)]
print(f"\n  Best K (silhouette score): {best_k_silhouette}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Elbow plot
axes[0].plot(k_range, inertias, 'o-', linewidth=2, markersize=8)
axes[0].axvline(x=elbow_k, color='r', linestyle='--', label=f'Elbow at K={elbow_k}')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia (Within-cluster sum of squares)')
axes[0].set_title('Elbow Method')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Silhouette score
axes[1].plot(k_range, silhouette_scores, 's-', linewidth=2, markersize=8, color='green')
axes[1].axvline(x=best_k_silhouette, color='r', linestyle='--', 
                label=f'Best K={best_k_silhouette}')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score Method')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kmeans_elbow_method.png', dpi=150, bbox_inches='tight')
print("✅ Saved plot: kmeans_elbow_method.png")

# ============================================
# Part 3: Customer Segmentation Example
# ============================================
print("\n=== Part 3: Customer Segmentation ===")
print("Real-world application: Grouping customers by behavior")

# Create customer data
np.random.seed(42)
n_customers = 300

# Features: Annual spending, Frequency of visits
# High-value customers
high_value = np.random.randn(n_customers//3, 2) * [500, 2] + [5000, 15]

# Medium-value customers
medium_value = np.random.randn(n_customers//3, 2) * [300, 1.5] + [2000, 8]

# Low-value customers
low_value = np.random.randn(n_customers - 2*(n_customers//3), 2) * [200, 1] + [500, 3]

# Combine
customer_data = np.vstack([high_value, medium_value, low_value])

# Create DataFrame
customers_df = pd.DataFrame(customer_data, columns=['annual_spending', 'visit_frequency'])

# Scale features (important for K-Means!)
scaler = StandardScaler()
customer_data_scaled = scaler.fit_transform(customer_data)

# Apply K-Means
kmeans_customers = KMeans(n_clusters=3, random_state=42, n_init=10)
customer_clusters = kmeans_customers.fit_predict(customer_data_scaled)

customers_df['cluster'] = customer_clusters

# Analyze clusters
print(f"\nCustomer Segmentation Results:")
print(f"\nCluster Statistics:")
for cluster_id in range(3):
    cluster_data = customers_df[customers_df['cluster'] == cluster_id]
    print(f"\n  Cluster {cluster_id}:")
    print(f"    Number of customers: {len(cluster_data)}")
    print(f"    Avg Annual Spending: ${cluster_data['annual_spending'].mean():.2f}")
    print(f"    Avg Visit Frequency: {cluster_data['visit_frequency'].mean():.2f}")

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(customers_df['annual_spending'], customers_df['visit_frequency'],
                     c=customers_df['cluster'], cmap='viridis', s=100, alpha=0.6)
centroids_scaled = kmeans_customers.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)
plt.scatter(centroids_original[:, 0], centroids_original[:, 1],
           c='red', marker='X', s=200, label='Centroids', 
           edgecolors='black', linewidths=2)
plt.xlabel('Annual Spending ($)')
plt.ylabel('Visit Frequency')
plt.title('Customer Segmentation (K-Means)')
plt.legend()
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)

# Cluster distribution
plt.subplot(1, 2, 2)
cluster_counts = customers_df['cluster'].value_counts().sort_index()
plt.bar(cluster_counts.index, cluster_counts.values, color=['green', 'blue', 'orange'])
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.title('Cluster Distribution')
plt.xticks(range(3))
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('kmeans_customer_segmentation.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved plot: kmeans_customer_segmentation.png")

# ============================================
# Part 4: K-Means Limitations
# ============================================
print("\n=== Part 4: K-Means Limitations ===")
print("K-Means assumes spherical clusters")

# Create non-spherical data (moon-shaped)
from sklearn.datasets import make_moons
X_moons, y_moons = make_moons(n_samples=200, noise=0.1, random_state=42)

# Try K-Means on non-spherical data
kmeans_moons = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_moons = kmeans_moons.fit_predict(X_moons)

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap='viridis', s=100, alpha=0.6)
plt.title('True Clusters (Non-spherical)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=labels_moons, cmap='viridis', s=100, alpha=0.6)
plt.scatter(kmeans_moons.cluster_centers_[:, 0], kmeans_moons.cluster_centers_[:, 1],
           c='red', marker='X', s=200, edgecolors='black', linewidths=2)
plt.title('K-Means Clustering (Fails on non-spherical data)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kmeans_limitations.png', dpi=150, bbox_inches='tight')
print("✅ Saved plot: kmeans_limitations.png")
print("\n→ K-Means struggles with non-spherical clusters!")
print("→ Consider DBSCAN or other algorithms for such data")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\n✅ K-Means: Groups data into K clusters")
print("✅ Unsupervised learning (no labels needed)")
print("✅ Finds cluster centroids")
print("\nKey Concepts:")
print("  - Inertia: Sum of squared distances to centroids")
print("  - Elbow Method: Find optimal K")
print("  - Silhouette Score: Measure cluster quality")
print("\nPros:")
print("  - Simple and fast")
print("  - Works well with spherical clusters")
print("  - Easy to interpret")
print("  - Scales to large datasets")
print("\nCons:")
print("  - Need to specify K (number of clusters)")
print("  - Assumes spherical clusters")
print("  - Sensitive to initialization")
print("  - Sensitive to outliers")
print("  - Requires feature scaling")
print("\nNext: Try pca.py (Dimensionality Reduction)!")

