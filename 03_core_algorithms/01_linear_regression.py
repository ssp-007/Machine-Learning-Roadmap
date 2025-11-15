"""
Linear Regression - Predicting Continuous Values
This tutorial covers:
1. Simple Linear Regression
2. Multiple Linear Regression
3. Polynomial Regression
4. Model evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("=" * 60)
print("LINEAR REGRESSION TUTORIAL")
print("=" * 60)

# ============================================
# Part 1: Simple Linear Regression
# ============================================
print("\n=== Part 1: Simple Linear Regression ===")
print("Predicting one variable from another (y = mx + b)")

# Create sample data: Hours studied vs Test score
np.random.seed(42)
hours_studied = np.random.randint(1, 50, 100)
# Test score = 50 + 2*hours + some noise
test_scores = 50 + 2 * hours_studied + np.random.randn(100) * 10

# Create DataFrame
data = pd.DataFrame({
    'hours_studied': hours_studied,
    'test_score': test_scores
})

# Prepare data
X = data[['hours_studied']]
y = data['test_score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"  Coefficient (slope): {model.coef_[0]:.2f}")
print(f"  Intercept: {model.intercept_:.2f}")
print(f"  R² Score: {r2:.4f}")
print(f"  RMSE: {rmse:.2f}")
print(f"  MAE: {mae:.2f}")

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, alpha=0.6, label='Training Data')
plt.plot(X_train, model.predict(X_train), 'r-', label='Model')
plt.xlabel('Hours Studied')
plt.ylabel('Test Score')
plt.title('Simple Linear Regression')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Test Score')
plt.ylabel('Predicted Test Score')
plt.title('Predicted vs Actual')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_regression_simple.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved plot: linear_regression_simple.png")

# ============================================
# Part 2: Multiple Linear Regression
# ============================================
print("\n=== Part 2: Multiple Linear Regression ===")
print("Predicting using multiple features")

# Create data with multiple features
np.random.seed(42)
n_samples = 200
data_multi = pd.DataFrame({
    'hours_studied': np.random.randint(1, 50, n_samples),
    'sleep_hours': np.random.randint(5, 10, n_samples),
    'exercise_hours': np.random.randint(0, 5, n_samples),
    'test_score': 0  # Will calculate
})

# Test score depends on multiple factors
data_multi['test_score'] = (
    30 + 
    2 * data_multi['hours_studied'] + 
    5 * data_multi['sleep_hours'] + 
    3 * data_multi['exercise_hours'] + 
    np.random.randn(n_samples) * 8
)

# Prepare data
X_multi = data_multi[['hours_studied', 'sleep_hours', 'exercise_hours']]
y_multi = data_multi['test_score']

# Split
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

# Train model
model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)

# Predict
y_pred_m = model_multi.predict(X_test_m)

# Evaluate
r2_multi = r2_score(y_test_m, y_pred_m)
rmse_multi = np.sqrt(mean_squared_error(y_test_m, y_pred_m))

print(f"\nModel Performance:")
print(f"  Coefficients:")
for i, feature in enumerate(X_multi.columns):
    print(f"    {feature}: {model_multi.coef_[i]:.2f}")
print(f"  Intercept: {model_multi.intercept_:.2f}")
print(f"  R² Score: {r2_multi:.4f}")
print(f"  RMSE: {rmse_multi:.2f}")

# ============================================
# Part 3: Polynomial Regression
# ============================================
print("\n=== Part 3: Polynomial Regression ===")
print("For non-linear relationships")

# Create non-linear data
np.random.seed(42)
X_poly = np.random.rand(100, 1) * 10
# y = x² + noise (quadratic relationship)
y_poly = 2 * X_poly.flatten()**2 + 3 * X_poly.flatten() + 5 + np.random.randn(100) * 10

# Split
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_poly, y_poly, test_size=0.2, random_state=42
)

# Transform features to polynomial
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train_p)
X_test_poly = poly_features.transform(X_test_p)

# Train polynomial model
model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train_p)

# Predict
y_pred_p = model_poly.predict(X_test_poly)

# Evaluate
r2_poly = r2_score(y_test_p, y_pred_p)
rmse_poly = np.sqrt(mean_squared_error(y_test_p, y_pred_p))

print(f"\nPolynomial Regression (degree=2) Performance:")
print(f"  R² Score: {r2_poly:.4f}")
print(f"  RMSE: {rmse_poly:.2f}")

# Compare with linear model
model_linear_p = LinearRegression()
model_linear_p.fit(X_train_p, y_train_p)
y_pred_linear_p = model_linear_p.predict(X_test_p)
r2_linear_p = r2_score(y_test_p, y_pred_linear_p)

print(f"\nLinear Regression (for comparison):")
print(f"  R² Score: {r2_linear_p:.4f}")
print(f"  → Polynomial is better for non-linear data!")

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
X_plot = np.linspace(X_poly.min(), X_poly.max(), 100).reshape(-1, 1)
X_plot_poly = poly_features.transform(X_plot)
y_plot = model_poly.predict(X_plot_poly)

plt.scatter(X_train_p, y_train_p, alpha=0.6, label='Training Data')
plt.plot(X_plot, y_plot, 'r-', label='Polynomial Model', linewidth=2)
plt.plot(X_plot, model_linear_p.predict(X_plot), 'g--', label='Linear Model', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial vs Linear Regression')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y_test_p, y_pred_p, alpha=0.6, label='Polynomial')
plt.scatter(y_test_p, y_pred_linear_p, alpha=0.6, label='Linear', marker='x')
plt.plot([y_test_p.min(), y_test_p.max()], [y_test_p.min(), y_test_p.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Predicted vs Actual')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('polynomial_regression.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved plot: polynomial_regression.png")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\n✅ Simple Linear Regression: One feature → one target")
print("✅ Multiple Linear Regression: Multiple features → one target")
print("✅ Polynomial Regression: Captures non-linear relationships")
print("\nKey Metrics:")
print("  - R² Score: Closer to 1.0 is better (explained variance)")
print("  - RMSE: Lower is better (error in same units as target)")
print("  - MAE: Lower is better (average error)")
print("\nNext: Try logistic_regression.py for classification!")

