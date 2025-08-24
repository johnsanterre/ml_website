# Principal Component Analysis (PCA) - 15 Minute Lecture Outline

## Learning Objectives
By the end of this lecture, students will understand:
- The curse of dimensionality and motivation for dimensionality reduction
- Mathematical foundations of PCA including eigendecomposition
- The geometric interpretation of principal components
- How to implement and interpret PCA results
- Practical considerations for choosing number of components

## 1. Introduction and Motivation (2 minutes)
- **The Curse of Dimensionality**
  - Exponential growth of volume with dimensions
  - Sparsity in high-dimensional spaces
  - Distance concentration phenomenon
- **Real-world Examples**
  - Image data: 28×28 pixel images = 784 dimensions
  - Gene expression data: thousands of genes
  - Text data: vocabulary size dimensions
- **Goals of Dimensionality Reduction**
  - Noise reduction and denoising
  - Visualization (2D/3D projections)
  - Computational efficiency
  - Storage compression

## 2. Mathematical Foundation (3 minutes)
- **Variance and Covariance**
  - Sample covariance matrix: $C = \frac{1}{n-1}X^TX$ (centered data)
  - Covariance captures linear relationships between features
- **Eigendecomposition**
  - Covariance matrix: $C = Q\Lambda Q^T$
  - Eigenvalues $\lambda_i$ represent variance along principal components
  - Eigenvectors $q_i$ represent directions of maximum variance
- **Principal Components**
  - First PC: direction of maximum variance
  - Subsequent PCs: orthogonal directions of decreasing variance
  - Mathematical constraint: $||q_i|| = 1$ and $q_i^T q_j = 0$ for $i \neq j$

## 3. Geometric Interpretation (2.5 minutes)
- **Data Cloud Visualization**
  - Original data in feature space
  - Principal components as new coordinate system
- **Variance Maximization**
  - PC1: line through data with maximum variance
  - PC2: perpendicular line with next highest variance
- **Dimensionality Reduction**
  - Projection onto first k principal components
  - Loss of information in discarded components
- **Reconstruction**
  - Approximate original data: $\tilde{X} = XQ_kQ_k^T$
  - Reconstruction error: $||X - \tilde{X}||_F^2$

## 4. PCA Algorithm (2 minutes)
- **Data Preprocessing**
  - Center the data: $X_{centered} = X - \mu$
  - Optional scaling: standardize features to unit variance
- **Eigendecomposition Steps**
  1. Compute covariance matrix $C = \frac{1}{n-1}X_{centered}^TX_{centered}$
  2. Find eigenvalues and eigenvectors: $C = Q\Lambda Q^T$
  3. Sort eigenvalues in descending order
  4. Select first k eigenvectors as principal components
- **Transformation**
  - Project data: $Y = X_{centered}Q_k$
  - New coordinates in principal component space

## 5. Choosing Number of Components (2 minutes)
- **Explained Variance Ratio**
  - Proportion of variance: $\frac{\lambda_i}{\sum_{j=1}^d \lambda_j}$
  - Cumulative explained variance
- **Scree Plot Analysis**
  - Plot eigenvalues vs component number
  - Look for "elbow" in the curve
- **Practical Rules**
  - 80-95% cumulative variance threshold
  - Kaiser criterion: keep components with $\lambda_i > 1$ (for standardized data)
  - Cross-validation for downstream tasks

## 6. Implementation Considerations (2 minutes)
- **Computational Approaches**
  - Full eigendecomposition: $O(d^3)$ complexity
  - SVD approach: $X = U\Sigma V^T$, components are columns of $V$
  - Randomized PCA for large datasets
- **Scaling Decisions**
  - Standardization when features have different units/scales
  - Mean centering always required
- **Interpretation Challenges**
  - Principal components are linear combinations of original features
  - Components may not have intuitive meaning
  - Sign ambiguity in eigenvectors

## 7. Practical Example and Visualization (1.5 minutes)
- **2D Example**
  - Correlated bivariate data
  - Show original axes vs principal component axes
  - Demonstrate dimensionality reduction to 1D
- **Real Data Application**
  - Iris dataset: 4D → 2D visualization
  - Show class separation in PC space
  - Compare to original feature pairs

## Key Takeaways
- **PCA finds orthogonal directions of maximum variance in data**
- **Eigenvalues quantify importance of each principal component**
- **Trade-off between dimensionality reduction and information loss**
- **Preprocessing (centering/scaling) significantly affects results**
- **Most useful for visualization, noise reduction, and computational efficiency**
- **Linear method - may miss nonlinear patterns in data**
