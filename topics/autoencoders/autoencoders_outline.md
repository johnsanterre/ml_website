# Autoencoders - 15 Minute Lecture Outline

## Learning Objectives
By the end of this lecture, students will understand:
- The architecture and motivation for autoencoders
- How autoencoders learn compressed representations
- Different types of autoencoders and their applications
- The relationship between autoencoders and dimensionality reduction
- Training process and loss functions for autoencoders

## 1. Introduction and Motivation (2 minutes)
- **What are Autoencoders?**
  - Neural networks that learn to copy their input to their output
  - Unsupervised learning approach for representation learning
  - Goal: Learn efficient coding of input data
- **Key Insight**
  - Bottleneck layer forces compression
  - Network learns to extract meaningful features
  - Reconstruction quality indicates representation quality
- **Applications**
  - Dimensionality reduction
  - Data denoising
  - Feature learning
  - Anomaly detection
  - Data generation

## 2. Basic Architecture (2.5 minutes)
- **Encoder Network**
  - Maps input $x \in \mathbb{R}^d$ to latent representation $z \in \mathbb{R}^k$
  - Function: $z = f_\theta(x)$ where $k < d$ (compression)
  - Typically uses ReLU, sigmoid, or tanh activations
- **Decoder Network**
  - Maps latent code back to reconstruction $\hat{x} \in \mathbb{R}^d$
  - Function: $\hat{x} = g_\phi(z)$ 
  - Mirror architecture of encoder (often symmetric)
- **Bottleneck Layer**
  - Smallest layer in the network (dimensionality $k$)
  - Forces information compression
  - Key hyperparameter: bottleneck size vs. reconstruction quality

## 3. Loss Functions and Training (2 minutes)
- **Reconstruction Loss**
  - Mean Squared Error: $L = \frac{1}{n}\sum_{i=1}^{n}||x_i - \hat{x}_i||^2$
  - Binary Cross-Entropy: $L = -\sum_{i=1}^{d}[x_i \log(\hat{x}_i) + (1-x_i)\log(1-\hat{x}_i)]$
  - Choice depends on input data type (continuous vs. binary)
- **Training Process**
  - Standard backpropagation through both encoder and decoder
  - No labels required (unsupervised learning)
  - Gradient flows: Input → Encoder → Bottleneck → Decoder → Reconstruction Loss
- **Optimization Challenges**
  - Avoiding trivial solutions (identity mapping)
  - Balancing compression vs. reconstruction fidelity
  - Preventing overfitting to training data

## 4. Types of Autoencoders (3 minutes)
- **Undercomplete Autoencoders**
  - Bottleneck smaller than input: $k < d$
  - Forces dimensionality reduction
  - Risk: May learn trivial features if capacity too limited
- **Overcomplete Autoencoders**
  - Bottleneck larger than input: $k > d$
  - Requires regularization to prevent copying
  - Techniques: Weight decay, sparsity constraints, noise injection
- **Sparse Autoencoders**
  - Add sparsity penalty: $L = L_{reconstruction} + \lambda \sum_j |h_j|$
  - Encourages few active neurons in hidden layers
  - Learns distributed but sparse representations
- **Denoising Autoencoders (DAE)**
  - Input: Corrupted data $\tilde{x} = x + \epsilon$
  - Output: Clean reconstruction $\hat{x} \approx x$
  - Forces network to learn robust features
  - Corruption types: Gaussian noise, masking, salt-and-pepper

## 5. Advanced Variants (2.5 minutes)
- **Variational Autoencoders (VAE)**
  - Probabilistic approach to latent representation
  - Encoder outputs parameters of latent distribution: $q_\phi(z|x)$
  - Loss includes KL divergence term: $L = L_{reconstruction} + \beta D_{KL}(q_\phi(z|x)||p(z))$
  - Enables generation of new samples
- **Convolutional Autoencoders**
  - Use CNN layers for encoder/decoder
  - Preserve spatial structure in images
  - Decoder uses transposed convolutions (deconvolutions)
  - Better for image data than fully connected layers
- **Sequence Autoencoders**
  - Use RNN/LSTM for sequential data
  - Encoder: Sequence → Fixed-size vector
  - Decoder: Vector → Reconstructed sequence
  - Applications: Text compression, sequence generation

## 6. Comparison with Other Methods (2 minutes)
- **Autoencoders vs. PCA**
  - PCA: Linear transformation, closed-form solution
  - Autoencoders: Non-linear, requires iterative training
  - Autoencoders can capture complex manifold structures
  - PCA guarantees optimal linear reconstruction
- **Autoencoders vs. t-SNE/UMAP**
  - t-SNE/UMAP: Designed for visualization (2D/3D)
  - Autoencoders: Flexible dimensionality, invertible transformation
  - t-SNE/UMAP better preserve local neighborhood structure
  - Autoencoders enable reconstruction and generation
- **When to Use Autoencoders**
  - Non-linear dimensionality reduction needed
  - Require invertible transformation
  - Large datasets where iterative training is feasible
  - Need feature learning for downstream tasks

## 7. Practical Applications (1 minute)
- **Image Compression and Denoising**
  - JPEG-style compression with learned features
  - Medical image denoising
  - Super-resolution reconstruction
- **Anomaly Detection**
  - Train on normal data only
  - High reconstruction error indicates anomaly
  - Applications: Fraud detection, system monitoring
- **Feature Learning**
  - Pre-train encoder on large unlabeled dataset
  - Use learned features for supervised downstream tasks
  - Transfer learning in domains with limited labeled data
- **Data Generation**
  - Sample from latent space and decode
  - Variational autoencoders for controllable generation
  - Data augmentation for training

## Key Takeaways
- **Autoencoders learn compressed representations through reconstruction**
- **Bottleneck architecture forces feature extraction and dimensionality reduction**
- **Different variants address specific challenges: sparsity, noise robustness, generation**
- **Non-linear alternative to PCA with greater flexibility but higher computational cost**
- **Training requires careful balance between compression and reconstruction quality**
- **Wide applications from compression to anomaly detection to generative modeling**
