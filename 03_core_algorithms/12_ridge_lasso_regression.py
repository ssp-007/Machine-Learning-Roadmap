"""
Ridge and Lasso Regression - Regularized Linear Regression
This tutorial covers:
1. Ridge Regression (L2 regularization)
2. Lasso Regression (L1 regularization)
3. Elastic Net (combination)
4. Regularization and overfitting
5. Feature selection with Lasso
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

print("=" * 60)
print("RIDGE AND LASSO REGRESSION TUTORIAL")
print("=" * 60)

# ============================================
# Part 1: Understanding Regularization
# ============================================
print("\n=== Part 1: Understanding Regularization ===")
print("Regularization prevents overfitting by penalizing large coefficients")

# Create data with many features (some irrelevant)
np.random.seed(42)
n_samples = 100
n_features = 20

# Only first 5 features are actually important
X = np.random.randn(n_samples, n_features)
# Target depends only on first 5 features
y = (X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] * 1.0 + 
     X[:, 3] * 0.8 + X[:, 4] * 0.5 + np.random.randn(n_samples) * 0.5)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (important for regularization!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nData Info:")
print(f"  Samples: {n_samples}")
print(f"  Features: {n_features}")
print(f"  Important features: First 5")
print(f"  Irrelevant features: Last 15")

# ============================================
# Part 2: Linear Regression (Baseline)
# ============================================
print("\n=== Part 2: Linear Regression (No Regularization) ===")

# Train regular linear regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Predictions
y_pred_lr = lr.predict(X_test_scaled)

# Evaluate
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"\nLinear Regression Performance:")
print(f"  RMSE: {rmse_lr:.4f}")
print(f"  R² Score: {r2_lr:.4f}")

# Show coefficients
coefficients_lr = pd.DataFrame({
    'feature': [f'Feature {i+1}' for i in range(n_features)],
    'coefficient': lr.coef_
}).sort_values('coefficient', key=abs, ascending=False)

print(f"\nTop 5 Coefficients (by absolute value):")
print(coefficients_lr.head())

# ============================================
# Part 3: Ridge Regression (L2 Regularization)
# ============================================
print("\n=== Part 3: Ridge Regression (L2 Regularization) ===")
print("Penalizes sum of squared coefficients (shrinks coefficients)")

# Try different alpha values
alphas_ridge = [0.01, 0.1, 1.0, 10.0, 100.0]
ridge_scores_train = []
ridge_scores_test = []
ridge_coefficients = []

for alpha in alphas_ridge:
    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    
    train_score = ridge.score(X_train_scaled, y_train)
    test_score = ridge.score(X_test_scaled, y_test)
    
    ridge_scores_train.append(train_score)
    ridge_scores_test.append(test_score)
    ridge_coefficients.append(ridge.coef_)

# Find best alpha
best_alpha_idx = np.argmax(ridge_scores_test)
best_alpha_ridge = alphas_ridge[best_alpha_idx]

print(f"\nBest Alpha: {best_alpha_ridge}")
print(f"  Training R²: {ridge_scores_train[best_alpha_idx]:.4f}")
print(f"  Test R²: {ridge_scores_test[best_alpha_idx]:.4f}")

# Train with best alpha
ridge_best = Ridge(alpha=best_alpha_ridge, random_state=42)
ridge_best.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_best.predict(X_test_scaled)

rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
print(f"  RMSE: {rmse_ridge:.4f}")

# Visualize effect of alpha
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(alphas_ridge, ridge_scores_train, 'o-', label='Training R²', linewidth=2)
plt.plot(alphas_ridge, ridge_scores_test, 's-', label='Test R²', linewidth=2)
plt.axvline(x=best_alpha_ridge, color='r', linestyle='--', label=f'Best α={best_alpha_ridge}')
plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('R² Score')
plt.title('Ridge Regression: Effect of Alpha')
plt.legend()
plt.grid(True, alpha=0.3)

# Coefficient shrinkage
plt.subplot(1, 2, 2)
for i, alpha in enumerate([0.01, 1.0, 100.0]):
    plt.plot(range(n_features), ridge_coefficients[i], 
            marker='o', label=f'α={alpha}', alpha=0.7)
plt.plot(range(n_features), lr.coef_, 'k--', label='No Regularization', linewidth=2)
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Coefficient Shrinkage (Ridge)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ridge_regression.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved plot: ridge_regression.png")

# ============================================
# Part 4: Lasso Regression (L1 Regularization)
# ============================================
print("\n=== Part 4: Lasso Regression (L1 Regularization) ===")
print("Penalizes sum of absolute coefficients (can zero out features)")

# Try different alpha values
alphas_lasso = [0.001, 0.01, 0.1, 1.0, 10.0]
lasso_scores_train = []
lasso_scores_test = []
lasso_coefficients = []
n_features_selected = []

for alpha in alphas_lasso:
    lasso = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    
    train_score = lasso.score(X_train_scaled, y_train)
    test_score = lasso.score(X_test_scaled, y_test)
    
    lasso_scores_train.append(train_score)
    lasso_scores_test.append(test_score)
    lasso_coefficients.append(lasso.coef_)
    n_features_selected.append(np.sum(lasso.coef_ != 0))

# Find best alpha
best_alpha_idx_lasso = np.argmax(lasso_scores_test)
best_alpha_lasso = alphas_lasso[best_alpha_idx_lasso]

print(f"\nBest Alpha: {best_alpha_lasso}")
print(f"  Training R²: {lasso_scores_train[best_alpha_idx_lasso]:.4f}")
print(f"  Test R²: {lasso_scores_test[best_alpha_idx_lasso]:.4f}")
print(f"  Features selected: {n_features_selected[best_alpha_idx_lasso]}/{n_features}")

# Train with best alpha
lasso_best = Lasso(alpha=best_alpha_lasso, random_state=42, max_iter=10000)
lasso_best.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_best.predict(X_test_scaled)

rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
print(f"  RMSE: {rmse_lasso:.4f}")

# Show selected features
selected_features = np.where(lasso_best.coef_ != 0)[0]
print(f"\nSelected Features: {selected_features + 1}")
print(f"  (Should be close to [1, 2, 3, 4, 5])")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Effect of alpha
axes[0].plot(alphas_lasso, lasso_scores_train, 'o-', label='Training R²', linewidth=2)
axes[0].plot(alphas_lasso, lasso_scores_test, 's-', label='Test R²', linewidth=2)
axes[0].axvline(x=best_alpha_lasso, color='r', linestyle='--', label=f'Best α={best_alpha_lasso}')
axes[0].set_xscale('log')
axes[0].set_xlabel('Alpha')
axes[0].set_ylabel('R² Score')
axes[0].set_title('Lasso: Effect of Alpha')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Coefficient paths
for i, alpha in enumerate([0.01, 0.1, 1.0]):
    axes[1].plot(range(n_features), lasso_coefficients[i], 
                marker='o', label=f'α={alpha}', alpha=0.7)
axes[1].plot(range(n_features), lr.coef_, 'k--', label='No Regularization', linewidth=2)
axes[1].set_xlabel('Feature Index')
axes[1].set_ylabel('Coefficient Value')
axes[1].set_title('Coefficient Paths (Lasso)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Number of features selected
axes[2].plot(alphas_lasso, n_features_selected, 'o-', linewidth=2, color='green')
axes[2].set_xscale('log')
axes[2].set_xlabel('Alpha')
axes[2].set_ylabel('Number of Features Selected')
axes[2].set_title('Feature Selection (Lasso)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lasso_regression.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved plot: lasso_regression.png")

# ============================================
# Part 5: Elastic Net (Combination)
# ============================================
print("\n=== Part 5: Elastic Net (L1 + L2 Regularization) ===")
print("Combines benefits of both Ridge and Lasso")

# Elastic Net with different l1_ratio values
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
elastic_scores = []

for l1_ratio in l1_ratios:
    elastic = ElasticNet(alpha=0.1, l1_ratio=l1_ratio, random_state=42, max_iter=10000)
    elastic.fit(X_train_scaled, y_train)
    score = elastic.score(X_test_scaled, y_test)
    elastic_scores.append(score)
    print(f"  l1_ratio={l1_ratio:.1f}: R² = {score:.4f}")

best_l1_ratio = l1_ratios[np.argmax(elastic_scores)]
print(f"\nBest l1_ratio: {best_l1_ratio}")

# ============================================
# Part 6: Comparison
# ============================================
print("\n=== Part 6: Comparison of All Methods ===")

comparison = pd.DataFrame({
    'Method': ['Linear Regression', 'Ridge', 'Lasso', 'Elastic Net'],
    'RMSE': [
        rmse_lr,
        rmse_ridge,
        rmse_lasso,
        np.sqrt(mean_squared_error(y_test, 
            ElasticNet(alpha=0.1, l1_ratio=best_l1_ratio, random_state=42, max_iter=10000)
            .fit(X_train_scaled, y_train).predict(X_test_scaled)))
    ],
    'R² Score': [
        r2_lr,
        ridge_scores_test[best_alpha_idx],
        lasso_scores_test[best_alpha_idx_lasso],
        max(elastic_scores)
    ],
    'Features Used': [
        n_features,
        n_features,
        n_features_selected[best_alpha_idx_lasso],
        np.sum(ElasticNet(alpha=0.1, l1_ratio=best_l1_ratio, random_state=42, max_iter=10000)
               .fit(X_train_scaled, y_train).coef_ != 0)
    ]
})

print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(comparison.round(4))

# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].bar(comparison['Method'], comparison['R² Score'], color=['blue', 'green', 'orange', 'red'])
axes[0].set_ylabel('R² Score')
axes[0].set_title('R² Score Comparison')
axes[0].set_ylim([0, 1])
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(comparison['Method'], comparison['Features Used'], color=['blue', 'green', 'orange', 'red'])
axes[1].set_ylabel('Number of Features')
axes[1].set_title('Feature Selection Comparison')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('regularization_comparison.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved plot: regularization_comparison.png")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\n✅ Regularization: Prevents overfitting by penalizing large coefficients")
print("✅ Ridge (L2): Shrinks coefficients, keeps all features")
print("✅ Lasso (L1): Can zero out coefficients, performs feature selection")
print("✅ Elastic Net: Combines both (good when features are correlated)")
print("\nKey Differences:")
print("  Ridge:")
print("    - Shrinks coefficients toward zero")
print("    - Keeps all features")
print("    - Good when all features are relevant")
print("  Lasso:")
print("    - Can set coefficients to exactly zero")
print("    - Performs automatic feature selection")
print("    - Good when many features are irrelevant")
print("  Elastic Net:")
print("    - Combines Ridge and Lasso")
print("    - Good when features are correlated")
print("\nWhen to Use:")
print("  - High-dimensional data (many features)")
print("  - Multicollinearity (correlated features)")
print("  - Overfitting issues")
print("  - Need feature selection (use Lasso)")
print("\nPros:")
print("  - Prevents overfitting")
print("  - Handles multicollinearity")
print("  - Lasso provides feature selection")
print("  - Better generalization")
print("\nCons:")
print("  - Need to tune alpha parameter")
print("  - Lasso may remove important features if alpha too high")
print("  - Requires feature scaling")
print("\nNext: You've completed all core algorithms! Move to Phase 4: Deep Learning!")

