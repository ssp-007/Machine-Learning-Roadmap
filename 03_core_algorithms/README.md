# Phase 3: Core ML Algorithms

## Learning Objectives
- Master the most important supervised learning algorithms
- Understand unsupervised learning techniques
- Learn comprehensive model evaluation methods
- Build and compare multiple models on real datasets

## Topics Covered

### 1. Supervised Learning Algorithms

#### Regression (Predicting Numbers)
- **Linear Regression** - Simple linear relationships
- **Polynomial Regression** - Non-linear relationships
- **Ridge Regression** - L2 regularization (prevents overfitting)
- **Lasso Regression** - L1 regularization (feature selection)
- **Elastic Net** - Combination of Ridge and Lasso

#### Classification (Predicting Categories)
- **Logistic Regression** - Binary and multi-class classification
- **Naive Bayes** - Probabilistic classifier (fast, good for text)
- **Decision Trees** - Rule-based classification
- **Random Forest** - Ensemble of decision trees
- **K-Nearest Neighbors (KNN)** - Instance-based learning
- **Support Vector Machines (SVM)** - Maximum margin classifier

### 2. Unsupervised Learning
- **K-Means Clustering** - Grouping similar data points
- **Principal Component Analysis (PCA)** - Dimensionality reduction

### 3. Model Evaluation
- **Train/Test Split** - Basic validation
- **Cross-Validation** - More robust validation
- **Confusion Matrix** - Classification performance
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

## Files in This Phase

1. **01_linear_regression.py** - Linear and polynomial regression
2. **02_logistic_regression.py** - Classification with logistic regression
3. **03_decision_trees.py** - Decision tree classifier
4. **04_random_forest.py** - Random forest ensemble
5. **05_knn.py** - K-Nearest Neighbors
6. **06_svm.py** - Support Vector Machines
7. **07_kmeans_clustering.py** - K-Means clustering
8. **08_pca.py** - Principal Component Analysis
9. **09_model_evaluation.py** - Comprehensive evaluation techniques
10. **10_complete_pipeline.ipynb** - End-to-end ML pipeline
11. **11_naive_bayes.py** - Naive Bayes classifier (Gaussian, Multinomial, Bernoulli)
12. **12_ridge_lasso_regression.py** - Regularized regression (Ridge, Lasso, Elastic Net)

## Key Concepts

### Supervised Learning
- **Regression**: Predict continuous values (price, temperature, etc.)
- **Classification**: Predict discrete categories (spam/not spam, disease/no disease)

### Unsupervised Learning
- **Clustering**: Find hidden patterns/groups in data
- **Dimensionality Reduction**: Reduce features while keeping important information

### Model Evaluation
- **Training Error**: How well model fits training data
- **Test Error**: How well model generalizes to new data
- **Bias-Variance Tradeoff**: Balance between underfitting and overfitting

## When to Use Which Algorithm?

### For Regression:
- **Linear Regression**: Simple, interpretable, fast
- **Ridge Regression**: Prevents overfitting, handles multicollinearity
- **Lasso Regression**: Feature selection, sparse models
- **Elastic Net**: Best of both Ridge and Lasso
- **Random Forest**: Non-linear relationships, feature importance

### For Classification:
- **Logistic Regression**: Simple, interpretable, baseline
- **Naive Bayes**: Very fast, excellent for text classification
- **Decision Trees**: Interpretable, handles non-linear data
- **Random Forest**: High accuracy, handles missing data
- **KNN**: Simple, works well with small datasets
- **SVM**: Good for high-dimensional data, clear separation

### For Clustering:
- **K-Means**: Most common, fast, works well with spherical clusters

## Practice Projects

1. **Iris Flower Classification** - Multi-class classification
2. **Wine Quality Prediction** - Regression problem
3. **Customer Segmentation** - Clustering application
4. **Spam Email Detection** - Binary classification
5. **House Price Prediction** - Advanced regression

## Resources
- [Scikit-learn Algorithm Cheat Sheet](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
- [ML Algorithm Comparison](https://www.kaggle.com/getting-started/131455)

## Advanced Algorithms (Covered in Later Phases)
- **Gradient Boosting** (XGBoost, LightGBM) - Advanced ensemble methods
- **Neural Networks** - Deep learning (Phase 4)
- **DBSCAN** - Density-based clustering
- **Hierarchical Clustering** - Alternative clustering method
- **Support Vector Regression (SVR)** - SVM for regression

## Next Steps
After completing this phase, you'll be ready for:
- Phase 4: Deep Learning
- Advanced model tuning and hyperparameter optimization
- Real-world ML projects
- Exploring advanced ensemble methods (XGBoost, etc.)

