# Cross Validation: Model Selection and Performance Estimation

## Learning Objectives
- Understand the importance of proper model evaluation
- Learn different cross validation techniques and their trade-offs
- Understand the bias-variance decomposition in model evaluation
- Learn how to select optimal hyperparameters using cross validation
- Understand common pitfalls and best practices in model validation
- Learn when to use different validation strategies

## Outline (15-minute lecture)

### 1. Introduction to Cross Validation
- The fundamental problem: limited data
- Train/validation/test split rationale
- Why simple train-test splits are insufficient
- The need for robust performance estimation

### 2. The Overfitting Problem
- Model complexity vs. generalization
- Training error vs. test error
- The danger of data snooping
- Why we need unbiased performance estimates

### 3. K-Fold Cross Validation
- Basic k-fold procedure
- Mathematical formulation
- Choosing k: bias-variance trade-off
- Standard choice: k=5 or k=10

### 4. Cross Validation Variants
- Leave-One-Out Cross Validation (LOOCV)
- Stratified k-fold for classification
- Time series cross validation
- Group-based cross validation

### 5. Nested Cross Validation
- Inner loop: hyperparameter selection
- Outer loop: performance estimation
- Avoiding optimistic bias in model selection
- Computational cost vs. reliability

### 6. Evaluation Metrics
- Classification: accuracy, precision, recall, F1, AUC-ROC
- Regression: MAE, MSE, RMSE, R²
- Cross validation with different metrics
- Metric selection based on problem requirements

### 7. Statistical Considerations
- Confidence intervals for CV estimates
- Paired t-tests for model comparison
- Variance of CV estimates
- Multiple comparison corrections

### 8. Common Pitfalls and Best Practices
- Data leakage in preprocessing
- Temporal dependencies in time series
- Imbalanced datasets
- Computational considerations for large datasets

### 9. Practical Implementation
- Scikit-learn cross validation tools
- Custom validation strategies
- Parallelization and efficiency
- Integration with hyperparameter optimization

### 10. Key Takeaways
- Cross validation provides unbiased performance estimates
- Choice of CV strategy depends on data characteristics
- Proper validation prevents overfitting to test set
- Essential tool for reliable machine learning
