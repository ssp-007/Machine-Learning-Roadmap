"""
Random Forest - Ensemble of Decision Trees
This tutorial covers:
1. Random Forest Classifier
2. Random Forest Regressor
3. Feature importance
4. Comparing with single decision tree
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_squared_error, r2_score
)
import seaborn as sns

print("=" * 60)
print("RANDOM FOREST TUTORIAL")
print("=" * 60)

# ============================================
# Part 1: Random Forest Classifier
# ============================================
print("\n=== Part 1: Random Forest Classifier ===")
print("Multiple decision trees voting together")

# Create sample data: Customer churn prediction
np.random.seed(42)
n_samples = 500

# Features
age = np.random.randint(18, 80, n_samples)
monthly_charges = np.random.uniform(20, 100, n_samples)
contract_length = np.random.choice([1, 12, 24], n_samples)
support_calls = np.random.randint(0, 10, n_samples)
tenure = np.random.randint(1, 72, n_samples)

# Target: Churn (1) or Stay (0)
# More likely to churn: high charges, many support calls, short tenure
churn_prob = (
    0.1 + 
    (monthly_charges > 70) * 0.3 +
    (support_calls > 5) * 0.2 +
    (tenure < 12) * 0.2 +
    (contract_length == 1) * 0.15 +
    np.random.randn(n_samples) * 0.1
)
churn_prob = np.clip(churn_prob, 0, 1)
churned = (churn_prob > 0.5).astype(int)

# Create DataFrame
data = pd.DataFrame({
    'age': age,
    'monthly_charges': monthly_charges,
    'contract_length': contract_length,
    'support_calls': support_calls,
    'tenure': tenure,
    'churned': churned
})

# Prepare data
X = data[['age', 'monthly_charges', 'contract_length', 'support_calls', 'tenure']]
y = data['churned']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create and train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nRandom Forest Performance:")
print(f"  Accuracy: {accuracy_rf:.4f}")

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nFeature Importance:")
for _, row in feature_importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
print(f"\nConfusion Matrix:")
print(cm_rf)

# Classification Report
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['Stay', 'Churn']))

# ============================================
# Part 2: Compare with Single Decision Tree
# ============================================
print("\n=== Part 2: Random Forest vs Single Decision Tree ===")

# Train single decision tree
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

print(f"\nComparison:")
print(f"  Single Decision Tree Accuracy: {accuracy_dt:.4f}")
print(f"  Random Forest Accuracy:        {accuracy_rf:.4f}")
print(f"  Improvement:                   {accuracy_rf - accuracy_dt:.4f}")

# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Confusion Matrix - Decision Tree
cm_dt = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Single Decision Tree')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# Confusion Matrix - Random Forest
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Random Forest')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('random_forest_comparison.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved plot: random_forest_comparison.png")

# Feature Importance Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

dt_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=True)

rf_importance = feature_importance.sort_values('importance', ascending=True)

axes[0].barh(dt_importance['feature'], dt_importance['importance'])
axes[0].set_title('Single Decision Tree - Feature Importance')
axes[0].set_xlabel('Importance')

axes[1].barh(rf_importance['feature'], rf_importance['importance'])
axes[1].set_title('Random Forest - Feature Importance')
axes[1].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('random_forest_feature_importance.png', dpi=150, bbox_inches='tight')
print("✅ Saved plot: random_forest_feature_importance.png")

# ============================================
# Part 3: Random Forest Regressor
# ============================================
print("\n=== Part 3: Random Forest Regressor ===")

# Create regression data: House prices
np.random.seed(42)
n_samples = 300

# Features
size = np.random.randint(800, 3000, n_samples)
bedrooms = np.random.randint(1, 5, n_samples)
bathrooms = np.random.uniform(1, 3, n_samples)
age = np.random.randint(0, 50, n_samples)
location_score = np.random.uniform(1, 10, n_samples)

# Target: Price (non-linear relationship)
price = (
    50000 + 
    size * 100 + 
    bedrooms * 20000 + 
    bathrooms * 15000 - 
    age * 500 + 
    location_score * 10000 +
    np.random.randn(n_samples) * 20000
)

# Create DataFrame
data_reg = pd.DataFrame({
    'size': size,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'age': age,
    'location_score': location_score,
    'price': price
})

# Prepare data
X_reg = data_reg[['size', 'bedrooms', 'bathrooms', 'age', 'location_score']]
y_reg = data_reg['price']

# Split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Train model
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_reg.fit(X_train_r, y_train_r)

# Predict
y_pred_rf_reg = rf_reg.predict(X_test_r)

# Evaluate
mse = mean_squared_error(y_test_r, y_pred_rf_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_r, y_pred_rf_reg)

print(f"\nModel Performance:")
print(f"  R² Score: {r2:.4f}")
print(f"  RMSE: ${rmse:,.2f}")

# Feature Importance
feature_importance_reg = pd.DataFrame({
    'feature': X_reg.columns,
    'importance': rf_reg.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nFeature Importance:")
for _, row in feature_importance_reg.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test_r, y_pred_rf_reg, alpha=0.6)
plt.plot([y_test_r.min(), y_test_r.max()], 
         [y_test_r.min(), y_test_r.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Random Forest Regression: Predicted vs Actual')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.barh(feature_importance_reg['feature'], feature_importance_reg['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('random_forest_regression.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved plot: random_forest_regression.png")

# ============================================
# Part 4: Effect of Number of Trees
# ============================================
print("\n=== Part 4: Effect of Number of Trees ===")

n_trees_list = [1, 5, 10, 25, 50, 100, 200]
train_scores = []
test_scores = []

for n_trees in n_trees_list:
    rf_temp = RandomForestClassifier(n_estimators=n_trees, max_depth=10, random_state=42)
    rf_temp.fit(X_train, y_train)
    
    train_score = rf_temp.score(X_train, y_train)
    test_score = rf_temp.score(X_test, y_test)
    
    train_scores.append(train_score)
    test_scores.append(test_score)

# Visualize
plt.figure(figsize=(10, 6))
plt.plot(n_trees_list, train_scores, 'o-', label='Training Score', linewidth=2)
plt.plot(n_trees_list, test_scores, 's-', label='Test Score', linewidth=2)
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Random Forest: Effect of Number of Trees')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('random_forest_n_trees.png', dpi=150, bbox_inches='tight')
print("✅ Saved plot: random_forest_n_trees.png")
print("→ More trees generally improve performance (up to a point)")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\n✅ Random Forest: Ensemble of decision trees")
print("✅ Each tree votes, majority wins")
print("✅ More robust than single decision tree")
print("✅ Less prone to overfitting")
print("\nKey Parameters:")
print("  - n_estimators: Number of trees (more = better, but slower)")
print("  - max_depth: Maximum depth of each tree")
print("  - max_features: Features to consider for each split")
print("\nPros:")
print("  - High accuracy")
print("  - Handles missing values well")
print("  - Feature importance")
print("  - Less overfitting than single tree")
print("\nCons:")
print("  - Less interpretable than single tree")
print("  - Can be slow with many trees")
print("  - Requires more memory")
print("\nNext: Try knn.py (K-Nearest Neighbors)!")

