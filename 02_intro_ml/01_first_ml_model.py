"""
Your First Machine Learning Model!
This is a simple linear regression example to predict house prices
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ============================================
# Step 1: Create Sample Data
# ============================================
print("=== Creating Sample Data ===")

# Let's create simple data: house size (sqft) vs price
# In real projects, you'd load data from CSV files
np.random.seed(42)  # For reproducibility

# Generate sample data
house_sizes = np.random.randint(1000, 5000, 100)  # House sizes in sqft
# Price = 100 * size + some random noise
house_prices = 100 * house_sizes + np.random.randint(-20000, 20000, 100)

# Create a DataFrame (like a spreadsheet)
data = pd.DataFrame({
    'size_sqft': house_sizes,
    'price': house_prices
})

print(f"Dataset shape: {data.shape}")
print("\nFirst 5 rows:")
print(data.head())

# ============================================
# Step 2: Explore the Data
# ============================================
print("\n=== Exploring Data ===")
print(f"\nBasic Statistics:")
print(data.describe())

# Visualize the data
plt.figure(figsize=(10, 6))
plt.scatter(data['size_sqft'], data['price'], alpha=0.6)
plt.xlabel('House Size (sqft)')
plt.ylabel('Price ($)')
plt.title('House Size vs Price')
plt.grid(True)
plt.savefig('house_data.png')
print("\nSaved plot as 'house_data.png'")

# ============================================
# Step 3: Prepare Data for ML
# ============================================
print("\n=== Preparing Data ===")

# Features (X) - what we use to make predictions
X = data[['size_sqft']]

# Target (y) - what we want to predict
y = data['price']

# Split data into training and testing sets
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# ============================================
# Step 4: Create and Train the Model
# ============================================
print("\n=== Training the Model ===")

# Create a Linear Regression model
model = LinearRegression()

# Train the model (this is where the "learning" happens!)
model.fit(X_train, y_train)

print("Model trained successfully!")
print(f"Model coefficient: {model.coef_[0]:.2f}")
print(f"Model intercept: {model.intercept_:.2f}")

# ============================================
# Step 5: Make Predictions
# ============================================
print("\n=== Making Predictions ===")

# Predict on test data
y_pred = model.predict(X_test)

# Show some predictions vs actual
comparison = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred,
    'Difference': y_test.values - y_pred
})

print("\nSample Predictions:")
print(comparison.head(10))

# ============================================
# Step 6: Evaluate the Model
# ============================================
print("\n=== Evaluating the Model ===")

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f} (1.0 is perfect)")

# ============================================
# Step 7: Visualize Results
# ============================================
print("\n=== Visualizing Results ===")

plt.figure(figsize=(12, 5))

# Plot 1: Training data and line
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, alpha=0.6, label='Training Data')
plt.plot(X_train, model.predict(X_train), 'r-', label='Model Prediction')
plt.xlabel('House Size (sqft)')
plt.ylabel('Price ($)')
plt.title('Training Data and Model')
plt.legend()
plt.grid(True)

# Plot 2: Test predictions
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, alpha=0.6, label='Actual')
plt.scatter(X_test, y_pred, alpha=0.6, label='Predicted', color='red')
plt.xlabel('House Size (sqft)')
plt.ylabel('Price ($)')
plt.title('Test Predictions')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('model_results.png')
print("Saved results plot as 'model_results.png'")

# ============================================
# Step 8: Make a Prediction on New Data
# ============================================
print("\n=== Making Predictions on New Data ===")

# Predict price for a 2500 sqft house
new_house_size = [[2500]]
predicted_price = model.predict(new_house_size)[0]

print(f"For a {new_house_size[0][0]} sqft house:")
print(f"Predicted price: ${predicted_price:,.2f}")

# ============================================
# Congratulations!
# ============================================
print("\n" + "="*50)
print("🎉 Congratulations! You built your first ML model!")
print("="*50)
print("\nWhat you learned:")
print("1. How to prepare data for ML")
print("2. How to split data into train/test sets")
print("3. How to train a model")
print("4. How to make predictions")
print("5. How to evaluate model performance")
print("\nNext steps:")
print("- Try the Iris classification example")
print("- Experiment with different features")
print("- Try other algorithms (Decision Trees, Random Forest)")

