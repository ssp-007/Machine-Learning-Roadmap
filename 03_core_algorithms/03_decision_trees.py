"""
Decision Trees - Rule-based Classification and Regression
This tutorial covers:
1. Decision Tree Classifier
2. Decision Tree Regressor
3. Visualizing decision trees
4. Feature importance
5. Overfitting in decision trees
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_squared_error, r2_score
)
import seaborn as sns

print("=" * 60)
print("DECISION TREES TUTORIAL")
print("=" * 60)

# ============================================
# Part 1: Decision Tree Classifier
# ============================================
print("\n=== Part 1: Decision Tree Classifier ===")
print("Classification using if-else rules")

# Create sample data: Loan approval
np.random.seed(42)
n_samples = 200

# Features
age = np.random.randint(18, 65, n_samples)
income = np.random.randint(20000, 150000, n_samples)
credit_score = np.random.randint(300, 850, n_samples)

# Target: Loan approved (1) or not (0)
# Rules: Age > 25, Income > 50000, Credit Score > 650
approved = (
    ((age > 25) & (income > 50000) & (credit_score > 650)).astype(int) +
    np.random.choice([0, 1], n_samples, p=[0.1, 0.9]) * 
    ((age > 30) & (income > 40000)).astype(int)
)
approved = np.clip(approved, 0, 1)

# Create DataFrame
data = pd.DataFrame({
    'age': age,
    'income': income,
    'credit_score': credit_score,
    'approved': approved
})

# Prepare data
X = data[['age', 'income', 'credit_score']]
y = data['approved']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create and train model (with limited depth to prevent overfitting)
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Performance:")
print(f"  Accuracy: {accuracy:.4f}")

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nFeature Importance:")
for _, row in feature_importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(cm)

# Visualize Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=['Not Approved', 'Approved'],
          filled=True, rounded=True, fontsize=10)
plt.title('Decision Tree Visualization', fontsize=16)
plt.tight_layout()
plt.savefig('decision_tree_structure.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved plot: decision_tree_structure.png")

# ============================================
# Part 2: Overfitting in Decision Trees
# ============================================
print("\n=== Part 2: Overfitting Demonstration ===")

# Train models with different depths
depths = [1, 3, 5, 10, 20]
train_scores = []
test_scores = []

for depth in depths:
    model_temp = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model_temp.fit(X_train, y_train)
    
    train_score = model_temp.score(X_train, y_train)
    test_score = model_temp.score(X_test, y_test)
    
    train_scores.append(train_score)
    test_scores.append(test_score)
    
    print(f"Depth {depth:2d}: Train={train_score:.4f}, Test={test_score:.4f}")

# Visualize overfitting
plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, 'o-', label='Training Score', linewidth=2)
plt.plot(depths, test_scores, 's-', label='Test Score', linewidth=2)
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.title('Overfitting in Decision Trees')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('decision_tree_overfitting.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved plot: decision_tree_overfitting.png")
print("→ Notice: Training score increases, but test score decreases after depth 5")

# ============================================
# Part 3: Decision Tree Regressor
# ============================================
print("\n=== Part 3: Decision Tree Regressor ===")
print("Using decision trees for regression (predicting numbers)")

# Create regression data
np.random.seed(42)
n_samples = 200

# Features
hours = np.random.randint(1, 50, n_samples)
experience = np.random.randint(0, 10, n_samples)

# Target: Salary (non-linear relationship)
salary = (
    30000 + 
    hours * 500 + 
    experience * 2000 + 
    hours * experience * 50 + 
    np.random.randn(n_samples) * 5000
)

# Create DataFrame
data_reg = pd.DataFrame({
    'hours': hours,
    'experience': experience,
    'salary': salary
})

# Prepare data
X_reg = data_reg[['hours', 'experience']]
y_reg = data_reg['salary']

# Split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Train model
model_reg = DecisionTreeRegressor(max_depth=4, random_state=42)
model_reg.fit(X_train_r, y_train_r)

# Predict
y_pred_r = model_reg.predict(X_test_r)

# Evaluate
mse = mean_squared_error(y_test_r, y_pred_r)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_r, y_pred_r)

print(f"\nModel Performance:")
print(f"  R² Score: {r2:.4f}")
print(f"  RMSE: ${rmse:,.2f}")

# Feature Importance
feature_importance_reg = pd.DataFrame({
    'feature': X_reg.columns,
    'importance': model_reg.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nFeature Importance:")
for _, row in feature_importance_reg.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Visualize predictions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test_r, y_pred_r, alpha=0.6)
plt.plot([y_test_r.min(), y_test_r.max()], 
         [y_test_r.min(), y_test_r.max()], 'r--', lw=2)
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Decision Tree Regression: Predicted vs Actual')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
residuals = y_test_r - y_pred_r
plt.scatter(y_pred_r, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Salary')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('decision_tree_regression.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved plot: decision_tree_regression.png")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\n✅ Decision Trees: Easy to interpret, rule-based")
print("✅ Can handle non-linear relationships")
print("✅ Feature importance shows which features matter most")
print("⚠️  Prone to overfitting (use max_depth, min_samples_split)")
print("\nPros:")
print("  - Easy to understand and visualize")
print("  - No feature scaling needed")
print("  - Handles both numerical and categorical data")
print("\nCons:")
print("  - Can overfit easily")
print("  - Unstable (small data changes → different tree)")
print("  - Biased toward features with more levels")
print("\nNext: Try random_forest.py (ensemble of decision trees)!")

