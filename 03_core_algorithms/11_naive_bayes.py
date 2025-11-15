"""
Naive Bayes - Probabilistic Classification
This tutorial covers:
1. Naive Bayes Classifier
2. Different variants (Gaussian, Multinomial, Bernoulli)
3. When to use Naive Bayes
4. Comparison with other classifiers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
import seaborn as sns

print("=" * 60)
print("NAIVE BAYES TUTORIAL")
print("=" * 60)

# ============================================
# Part 1: Gaussian Naive Bayes
# ============================================
print("\n=== Part 1: Gaussian Naive Bayes ===")
print("For continuous features (assumes Gaussian distribution)")

# Create sample data: Email classification (Spam/Not Spam)
np.random.seed(42)
n_samples = 500

# Features: Word count, Link count, Exclamation marks
# Spam emails tend to have more links and exclamation marks
spam_word_count = np.random.normal(50, 15, n_samples//2)
spam_link_count = np.random.poisson(5, n_samples//2)
spam_exclamation = np.random.poisson(3, n_samples//2)

# Not spam emails
not_spam_word_count = np.random.normal(200, 50, n_samples//2)
not_spam_link_count = np.random.poisson(1, n_samples//2)
not_spam_exclamation = np.random.poisson(0.5, n_samples//2)

# Combine
word_count = np.concatenate([spam_word_count, not_spam_word_count])
link_count = np.concatenate([spam_link_count, not_spam_link_count])
exclamation = np.concatenate([spam_exclamation, not_spam_exclamation])
labels = np.concatenate([np.ones(n_samples//2), np.zeros(n_samples//2)])

# Create DataFrame
data = pd.DataFrame({
    'word_count': word_count,
    'link_count': link_count,
    'exclamation': exclamation,
    'is_spam': labels
})

# Prepare data
X = data[['word_count', 'link_count', 'exclamation']]
y = data['is_spam']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (Naive Bayes doesn't require it, but helps visualization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Make predictions
y_pred = gnb.predict(X_test)
y_pred_proba = gnb.predict_proba(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nGaussian Naive Bayes Performance:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1-Score: {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(cm)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Not Spam', 'Spam'],
            yticklabels=['Not Spam', 'Spam'])
axes[0].set_title('Confusion Matrix - Gaussian Naive Bayes')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# Probability distribution
axes[1].hist(y_pred_proba[y_test == 0, 1], bins=20, alpha=0.7, 
            label='Actually Not Spam', color='blue')
axes[1].hist(y_pred_proba[y_test == 1, 1], bins=20, alpha=0.7, 
            label='Actually Spam', color='red')
axes[1].axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
axes[1].set_xlabel('Predicted Probability of Spam')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Probability Distribution')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('naive_bayes_gaussian.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved plot: naive_bayes_gaussian.png")

# ============================================
# Part 2: Multinomial Naive Bayes
# ============================================
print("\n=== Part 2: Multinomial Naive Bayes ===")
print("For discrete counts (e.g., word counts in text)")

# Create text classification data (word counts)
np.random.seed(42)
n_samples = 300

# Features: Counts of specific words
# Spam: high "free", "click", "now" counts
# Not spam: high "meeting", "project", "team" counts
spam_data = {
    'free': np.random.poisson(5, n_samples//2),
    'click': np.random.poisson(4, n_samples//2),
    'now': np.random.poisson(3, n_samples//2),
    'meeting': np.random.poisson(0.5, n_samples//2),
    'project': np.random.poisson(0.3, n_samples//2),
    'team': np.random.poisson(0.2, n_samples//2)
}

not_spam_data = {
    'free': np.random.poisson(0.2, n_samples//2),
    'click': np.random.poisson(0.1, n_samples//2),
    'now': np.random.poisson(0.5, n_samples//2),
    'meeting': np.random.poisson(4, n_samples//2),
    'project': np.random.poisson(5, n_samples//2),
    'team': np.random.poisson(3, n_samples//2)
}

# Combine
X_multi = pd.DataFrame({
    'free': np.concatenate([spam_data['free'], not_spam_data['free']]),
    'click': np.concatenate([spam_data['click'], not_spam_data['click']]),
    'now': np.concatenate([spam_data['now'], not_spam_data['now']]),
    'meeting': np.concatenate([spam_data['meeting'], not_spam_data['meeting']]),
    'project': np.concatenate([spam_data['project'], not_spam_data['project']]),
    'team': np.concatenate([spam_data['team'], not_spam_data['team']])
})
y_multi = np.concatenate([np.ones(n_samples//2), np.zeros(n_samples//2)])

# Split
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42, stratify=y_multi
)

# Train Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train_m, y_train_m)

# Predict
y_pred_m = mnb.predict(X_test_m)

# Evaluate
accuracy_m = accuracy_score(y_test_m, y_pred_m)
print(f"\nMultinomial Naive Bayes Performance:")
print(f"  Accuracy: {accuracy_m:.4f}")

# ============================================
# Part 3: Bernoulli Naive Bayes
# ============================================
print("\n=== Part 3: Bernoulli Naive Bayes ===")
print("For binary features (word present/absent)")

# Convert word counts to binary (present/absent)
X_bernoulli = (X_multi > 0).astype(int)

# Split
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_bernoulli, y_multi, test_size=0.2, random_state=42, stratify=y_multi
)

# Train Bernoulli Naive Bayes
bnb = BernoulliNB()
bnb.fit(X_train_b, y_train_b)

# Predict
y_pred_b = bnb.predict(X_test_b)

# Evaluate
accuracy_b = accuracy_score(y_test_b, y_pred_b)
print(f"\nBernoulli Naive Bayes Performance:")
print(f"  Accuracy: {accuracy_b:.4f}")

# ============================================
# Part 4: Comparison with Other Algorithms
# ============================================
print("\n=== Part 4: Naive Bayes vs Other Algorithms ===")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Compare on original spam dataset
models_compare = {
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []

for name, model in models_compare.items():
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred_comp = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred_comp = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred_comp)
    results.append({'Model': name, 'Accuracy': acc})
    print(f"  {name}: {acc:.4f}")

results_df = pd.DataFrame(results)
print(f"\nComparison Results:")
print(results_df)

# Visualize
plt.figure(figsize=(10, 6))
plt.bar(results_df['Model'], results_df['Accuracy'], color=['blue', 'green', 'orange'])
plt.ylabel('Accuracy')
plt.title('Naive Bayes vs Other Classifiers')
plt.ylim([0, 1])
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('naive_bayes_comparison.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved plot: naive_bayes_comparison.png")

# ============================================
# Part 5: Understanding "Naive" Assumption
# ============================================
print("\n=== Part 5: Why is it 'Naive'? ===")
print("Naive Bayes assumes features are independent")
print("(This is rarely true in real life, but it still works well!)")

# Show feature correlation
correlation = X.corr()
print(f"\nFeature Correlations:")
print(correlation.round(3))

# Visualize correlations
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm', center=0)
plt.title('Feature Correlations')
plt.tight_layout()
plt.savefig('naive_bayes_correlations.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved plot: naive_bayes_correlations.png")
print("\n→ Even with correlations, Naive Bayes performs well!")
print("→ This is why it's called 'Naive' but still useful")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\n✅ Naive Bayes: Probabilistic classifier based on Bayes' theorem")
print("✅ Assumes features are independent (naive assumption)")
print("✅ Fast, simple, and works well despite the assumption")
print("\nVariants:")
print("  - GaussianNB: For continuous features")
print("  - MultinomialNB: For count data (word counts)")
print("  - BernoulliNB: For binary features")
print("\nPros:")
print("  - Very fast training and prediction")
print("  - Works well with small datasets")
print("  - Good baseline for text classification")
print("  - Handles multiple classes naturally")
print("  - Not sensitive to irrelevant features")
print("\nCons:")
print("  - Assumes feature independence (rarely true)")
print("  - Can be outperformed by more complex models")
print("  - Requires feature scaling for GaussianNB")
print("\nWhen to Use:")
print("  - Text classification (spam, sentiment)")
print("  - Quick baseline model")
print("  - Small datasets")
print("  - When interpretability matters")
print("\nNext: Try ridge_lasso_regression.py for regularized regression!")

