"""
Comprehensive Model Evaluation
This tutorial covers:
1. Train/Test Split
2. Cross-Validation
3. Confusion Matrix
4. Classification Metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
5. Regression Metrics (MSE, RMSE, MAE, R²)
6. Overfitting Detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score
)
import seaborn as sns

print("=" * 60)
print("COMPREHENSIVE MODEL EVALUATION TUTORIAL")
print("=" * 60)

# ============================================
# Part 1: Train/Test Split
# ============================================
print("\n=== Part 1: Train/Test Split ===")
print("Basic validation method")

# Create sample data
np.random.seed(42)
n_samples = 200

X = np.random.randn(n_samples, 3)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nData Split:")
print(f"  Total samples: {len(X)}")
print(f"  Training set: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Test set: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# Train model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"\nModel Performance:")
print(f"  Training Accuracy: {train_score:.4f}")
print(f"  Test Accuracy: {test_score:.4f}")
print(f"  Difference: {abs(train_score - test_score):.4f}")

if abs(train_score - test_score) > 0.1:
    print("  ⚠️  Large gap suggests overfitting!")
else:
    print("  ✅ Good generalization!")

# ============================================
# Part 2: Cross-Validation
# ============================================
print("\n=== Part 2: Cross-Validation ===")
print("More robust validation using K-fold CV")

# K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(f"\n5-Fold Cross-Validation Results:")
print(f"  Individual scores: {cv_scores}")
print(f"  Mean: {cv_scores.mean():.4f}")
print(f"  Std Dev: {cv_scores.std():.4f}")
print(f"  Min: {cv_scores.min():.4f}")
print(f"  Max: {cv_scores.max():.4f}")

# Stratified K-Fold (for classification with imbalanced classes)
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_stratified = cross_val_score(model, X, y, cv=skfold, scoring='accuracy')

print(f"\nStratified 5-Fold Cross-Validation:")
print(f"  Mean: {cv_scores_stratified.mean():.4f}")
print(f"  Std Dev: {cv_scores_stratified.std():.4f}")

# Visualize CV scores
plt.figure(figsize=(10, 6))
plt.boxplot([cv_scores, cv_scores_stratified], labels=['K-Fold', 'Stratified K-Fold'])
plt.ylabel('Accuracy')
plt.title('Cross-Validation Scores Comparison')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('cross_validation.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved plot: cross_validation.png")

# ============================================
# Part 3: Classification Metrics
# ============================================
print("\n=== Part 3: Classification Metrics ===")

# Create classification problem
np.random.seed(42)
n_samples = 300

X_clf = np.random.randn(n_samples, 4)
y_clf = (X_clf[:, 0] + X_clf[:, 1] > 0).astype(int)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# Train model
clf_model = RandomForestClassifier(random_state=42)
clf_model.fit(X_train_clf, y_train_clf)

# Predictions
y_pred_clf = clf_model.predict(X_test_clf)
y_pred_proba_clf = clf_model.predict_proba(X_test_clf)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test_clf, y_pred_clf)
precision = precision_score(y_test_clf, y_pred_clf)
recall = recall_score(y_test_clf, y_pred_clf)
f1 = f1_score(y_test_clf, y_pred_clf)
roc_auc = roc_auc_score(y_test_clf, y_pred_proba_clf)

print(f"\nClassification Metrics:")
print(f"  Accuracy:  {accuracy:.4f}  (Overall correctness)")
print(f"  Precision: {precision:.4f}  (Of predicted positives, how many are correct?)")
print(f"  Recall:    {recall:.4f}  (Of actual positives, how many did we catch?)")
print(f"  F1-Score:  {f1:.4f}  (Harmonic mean of precision and recall)")
print(f"  ROC-AUC:   {roc_auc:.4f}  (Area under ROC curve)")

# Confusion Matrix
cm = confusion_matrix(y_test_clf, y_pred_clf)
print(f"\nConfusion Matrix:")
print(f"                Predicted")
print(f"              Negative  Positive")
print(f"Actual Negative    {cm[0,0]:3d}      {cm[0,1]:3d}")
print(f"        Positive    {cm[1,0]:3d}      {cm[1,1]:3d}")

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved plot: confusion_matrix.png")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test_clf, y_pred_proba_clf)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
print("✅ Saved plot: roc_curve.png")

# Classification Report
print(f"\nDetailed Classification Report:")
print(classification_report(y_test_clf, y_pred_clf, 
                            target_names=['Negative', 'Positive']))

# ============================================
# Part 4: Regression Metrics
# ============================================
print("\n=== Part 4: Regression Metrics ===")

# Create regression problem
np.random.seed(42)
n_samples = 200

X_reg = np.random.randn(n_samples, 3)
y_reg = X_reg[:, 0] * 2 + X_reg[:, 1] * 1.5 + np.random.randn(n_samples) * 0.5

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Train model
reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)

# Predictions
y_pred_reg = reg_model.predict(X_test_reg)

# Calculate metrics
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"\nRegression Metrics:")
print(f"  MSE:  {mse:.4f}  (Mean Squared Error)")
print(f"  RMSE: {rmse:.4f}  (Root Mean Squared Error - in target units)")
print(f"  MAE:  {mae:.4f}  (Mean Absolute Error - average error)")
print(f"  R²:   {r2:.4f}  (Coefficient of Determination)")

# Visualize predictions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test_reg, y_pred_reg, alpha=0.6)
plt.plot([y_test_reg.min(), y_test_reg.max()], 
         [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Predicted vs Actual')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
residuals = y_test_reg - y_pred_reg
plt.scatter(y_pred_reg, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regression_metrics.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved plot: regression_metrics.png")

# ============================================
# Part 5: Overfitting Detection
# ============================================
print("\n=== Part 5: Overfitting Detection ===")

# Create data
np.random.seed(42)
X_overfit = np.random.randn(100, 5)
y_overfit = (X_overfit[:, 0] > 0).astype(int)

X_train_of, X_test_of, y_train_of, y_test_of = train_test_split(
    X_overfit, y_overfit, test_size=0.2, random_state=42, stratify=y_overfit
)

# Train models with different complexities
depths = [1, 3, 5, 10, 20]
train_scores = []
test_scores = []

for depth in depths:
    model_of = RandomForestClassifier(max_depth=depth, random_state=42)
    model_of.fit(X_train_of, y_train_of)
    
    train_score = model_of.score(X_train_of, y_train_of)
    test_score = model_of.score(X_test_of, y_test_of)
    
    train_scores.append(train_score)
    test_scores.append(test_score)

# Visualize overfitting
plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, 'o-', label='Training Score', linewidth=2)
plt.plot(depths, test_scores, 's-', label='Test Score', linewidth=2)
plt.xlabel('Model Complexity (Max Depth)')
plt.ylabel('Accuracy')
plt.title('Overfitting Detection')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('overfitting_detection.png', dpi=150, bbox_inches='tight')
print("✅ Saved plot: overfitting_detection.png")

print(f"\nObservations:")
best_idx = np.argmax(test_scores)
print(f"  Best test performance at depth {depths[best_idx]}")
print(f"  Training score keeps increasing (memorizing)")
print(f"  Test score decreases after depth {depths[best_idx]} (overfitting)")

# ============================================
# Part 6: Metric Comparison Summary
# ============================================
print("\n=== Part 6: When to Use Which Metric ===")

print("\nFor Classification:")
print("  - Accuracy: Good for balanced classes")
print("  - Precision: When false positives are costly (e.g., spam detection)")
print("  - Recall: When false negatives are costly (e.g., disease detection)")
print("  - F1-Score: Balance between precision and recall")
print("  - ROC-AUC: Overall model performance (threshold-independent)")

print("\nFor Regression:")
print("  - MSE/RMSE: Penalizes large errors more")
print("  - MAE: Average error, easier to interpret")
print("  - R²: Proportion of variance explained (0-1, higher is better)")

print("\nValidation Methods:")
print("  - Train/Test Split: Quick, simple")
print("  - Cross-Validation: More robust, better estimate")
print("  - Stratified CV: For imbalanced classes")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\n✅ Always split data into train/test sets")
print("✅ Use cross-validation for better estimates")
print("✅ Choose metrics based on your problem")
print("✅ Monitor for overfitting (train vs test performance)")
print("\nKey Takeaways:")
print("  - No single metric tells the whole story")
print("  - Consider your business/problem context")
print("  - Visualize results (confusion matrix, ROC curve, etc.)")
print("  - Cross-validation > single train/test split")
print("\nCongratulations! You've learned comprehensive model evaluation!")

