# CART Algorithm - Lecture Outline
## 15-Minute Lecture on Classification and Regression Trees

### Learning Objectives
- Understand the fundamental principles of the CART algorithm
- Learn how CART makes binary splitting decisions
- Master the mathematical formulation of impurity measures
- Comprehend the difference between classification and regression trees
- Apply CART concepts to real-world decision-making problems

### 1. Introduction to CART (2 minutes)
- **What is CART?** Classification and Regression Trees
- **Key Idea:** Recursive binary partitioning of feature space
- **Two Types:** Classification trees (categorical outcomes) vs Regression trees (continuous outcomes)
- **Advantages:** Interpretable, handles mixed data types, no assumptions about data distribution

### 2. Binary Splitting Mechanism (3 minutes)
- **Recursive Partitioning:** Start with all data, split into two subsets
- **Split Criteria:** Find the best feature and threshold that maximizes information gain
- **Stopping Conditions:** Maximum depth, minimum samples per leaf, minimum impurity decrease
- **Tree Structure:** Root node → internal nodes → leaf nodes

### 3. Impurity Measures (4 minutes)
- **Classification Trees:** Gini Impurity and Entropy
  - Gini Impurity: $G = 1 - \sum_{i=1}^{c} p_i^2$
  - Entropy: $H = -\sum_{i=1}^{c} p_i \log_2(p_i)$
- **Regression Trees:** Mean Squared Error (MSE)
  - MSE: $\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2$
- **Information Gain:** $\text{IG} = \text{Impurity(parent)} - \sum_{j=1}^{k} \frac{n_j}{n} \text{Impurity(child}_j)$

### 4. Algorithm Implementation (3 minutes)
- **Step 1:** Calculate impurity for current node
- **Step 2:** For each feature, find optimal split threshold
- **Step 3:** Select split with maximum information gain
- **Step 4:** Create child nodes and recurse
- **Step 5:** Assign predictions to leaf nodes

### 5. Practical Example (2 minutes)
- **Dataset:** Simple 2D classification problem
- **Visualization:** Show how CART partitions the space
- **Decision Path:** Follow a sample through the tree
- **Interpretation:** Explain the decision rules

### 6. Key Takeaways (1 minute)
- CART uses recursive binary splitting to partition feature space
- Impurity measures guide optimal split selection
- Different impurity measures for classification vs regression
- Trees provide interpretable decision rules
- Stopping criteria prevent overfitting

### Mathematical Foundation
- Binary splitting: $x_j \leq t$ vs $x_j > t$
- Information gain maximization
- Impurity minimization at each node
- Recursive partitioning until stopping criteria met

### Practical Considerations
- Feature scaling not required
- Handles missing values through surrogate splits
- Computationally efficient: $O(n \log n)$ per split
- Memory efficient for large datasets
