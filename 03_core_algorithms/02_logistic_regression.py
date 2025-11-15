"""
Logistic Regression - Binary and Multi-class Classification
This tutorial covers:
1. Binary Classification (Yes/No, Spam/Not Spam)
2. Multi-class Classification
3. Probability predictions
4. Model evaluation for classification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
import seaborn as sns

print("=" * 60)
print("LOGISTIC REGRESSION TUTORIAL")
print("=" * 60)

# ============================================
# Part 1: Binary Classification
# ============================================
print("\n=== Part 1: Binary Classification ===")
print("Predicting two classes (e.g., Pass/Fail, Spam/Not Spam)")

# Create sample data: Student performance
np.random.seed(42)
n_samples = 200

# Features
hours_studied = np.random.randint(1, 50, n_samples)
attendance = np.random.uniform(0.5, 1.0, n_samples) * 100

# Target: Pass (1) or Fail (0)
# Higher hours and attendance = more likely to pass
pass_probability = (
    0.3 + 
    0.01 * hours_studied + 
    0.005 * attendance + 
    np.random.randn(n_samples) * 0.1
)
pass_probability = np.clip(pass_probability, 0, 1)
passed = (pass_probability > 0.5).astype(int)

# Create DataFrame
data = pd.DataFrame({
    'hours_studied': hours_studied,
    'attendance': attendance,
    'passed': passed
})

# Prepare data
X = data[['hours_studied', 'attendance']]
y = data['passed']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create and train model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nModel Performance:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1-Score: {f1:.4f}")
print(f"  ROC-AUC: {roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(cm)

# Classification Report
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fail', 'Pass']))

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].set_xlabel('Predicted')

# 2. Decision Boundary (simplified 2D view)
h = 0.5
x_min, x_max = X['hours_studied'].min() - 1, X['hours_studied'].max() + 1
y_min, y_max = X['attendance'].min() - 1, X['attendance'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

axes[0, 1].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
scatter = axes[0, 1].scatter(X_test['hours_studied'], X_test['attendance'], 
                            c=y_test, cmap='RdYlBu', edgecolors='black')
axes[0, 1].set_xlabel('Hours Studied')
axes[0, 1].set_ylabel('Attendance (%)')
axes[0, 1].set_title('Decision Boundary')
plt.colorbar(scatter, ax=axes[0, 1])

# 3. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
axes[1, 0].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].set_title('ROC Curve')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Probability Distribution
axes[1, 1].hist(y_pred_proba[y_test == 0], bins=20, alpha=0.7, 
                label='Actually Failed', color='red')
axes[1, 1].hist(y_pred_proba[y_test == 1], bins=20, alpha=0.7, 
                label='Actually Passed', color='green')
axes[1, 1].axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
axes[1, 1].set_xlabel('Predicted Probability')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Probability Distribution')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('logistic_regression_binary.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved plot: logistic_regression_binary.png")

# ============================================
# Part 2: Multi-class Classification
# ============================================
print("\n=== Part 2: Multi-class Classification ===")
print("Predicting multiple classes (e.g., Grade A, B, C, D)")

# Create data for grade prediction
np.random.seed(42)
n_samples = 300

# Features
hours = np.random.randint(1, 50, n_samples)
attendance = np.random.uniform(0.5, 1.0, n_samples) * 100
assignments = np.random.randint(0, 20, n_samples)

# Calculate score
total_score = hours * 2 + attendance * 0.5 + assignments * 3 + np.random.randn(n_samples) * 10

# Assign grades based on score
def assign_grade(score):
    if score >= 180:
        return 'A'
    elif score >= 150:
        return 'B'
    elif score >= 120:
        return 'C'
    else:
        return 'D'

grades = [assign_grade(s) for s in total_score]

# Create DataFrame
data_multi = pd.DataFrame({
    'hours': hours,
    'attendance': attendance,
    'assignments': assignments,
    'grade': grades
})

# Prepare data
X_multi = data_multi[['hours', 'attendance', 'assignments']]
y_multi = data_multi['grade']

# Split
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42, stratify=y_multi
)

# Train model
model_multi = LogisticRegression(random_state=42, max_iter=1000, multi_class='multinomial')
model_multi.fit(X_train_m, y_train_m)

# Predict
y_pred_m = model_multi.predict(X_test_m)

# Evaluate
accuracy_multi = accuracy_score(y_test_m, y_pred_m)

print(f"\nModel Performance:")
print(f"  Accuracy: {accuracy_multi:.4f}")

# Confusion Matrix
cm_multi = confusion_matrix(y_test_m, y_pred_m, labels=['A', 'B', 'C', 'D'])

print(f"\nConfusion Matrix:")
print(pd.DataFrame(cm_multi, 
                   index=['Actual A', 'Actual B', 'Actual C', 'Actual D'],
                   columns=['Pred A', 'Pred B', 'Pred C', 'Pred D']))

# Classification Report
print(f"\nClassification Report:")
print(classification_report(y_test_m, y_pred_m, target_names=['A', 'B', 'C', 'D']))

# Visualize Multi-class Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['A', 'B', 'C', 'D'],
            yticklabels=['A', 'B', 'C', 'D'])
plt.title('Multi-class Confusion Matrix')
plt.ylabel('Actual Grade')
plt.xlabel('Predicted Grade')
plt.tight_layout()
plt.savefig('logistic_regression_multiclass.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved plot: logistic_regression_multiclass.png")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\n✅ Binary Classification: Two classes (0/1, Yes/No)")
print("✅ Multi-class Classification: Multiple classes (A/B/C/D)")
print("\nKey Metrics:")
print("  - Accuracy: Overall correctness")
print("  - Precision: Of predicted positives, how many are actually positive?")
print("  - Recall: Of actual positives, how many did we catch?")
print("  - F1-Score: Harmonic mean of precision and recall")
print("  - ROC-AUC: Area under ROC curve (binary only)")
print("\nNext: Try decision_trees.py!")

