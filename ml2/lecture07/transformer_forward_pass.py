"""
Pure NumPy implementation of a Transformer forward pass
No training - just demonstrates the matrix math with fixed weights
"""

import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Hyperparameters
d_model = 4        # Embedding dimension
n_heads = 2         # Number of attention heads
d_k = d_model // n_heads  # Dimension per head (8)
d_ff = 64           # Feed-forward hidden dimension
seq_len = 4         # Sequence length
batch_size = 2      # Number of examples
vocab_size = 100    # Vocabulary size

print("="*60)
print("TRANSFORMER FORWARD PASS - PURE NUMPY")
print("="*60)
print(f"Config: d_model={d_model}, n_heads={n_heads}, d_k={d_k}")
print(f"        seq_len={seq_len}, batch_size={batch_size}")
print("="*60)


# ============================================================================
# 1. INPUT: Token IDs and Embeddings
# ============================================================================
print("\n1. INPUT EMBEDDINGS")
print("-"*60)

# Input token IDs (batch_size, seq_len)
token_ids = np.array([
    [5, 12, 23, 7],
    [8, 15, 3, 19]
])
print(f"Token IDs shape: {token_ids.shape}")
print(f"Token IDs:\n{token_ids}")

# Embedding matrix (vocab_size, d_model)
embedding_matrix = np.random.randn(vocab_size, d_model) * 0.1

# Look up embeddings (batch_size, seq_len, d_model)
X = embedding_matrix[token_ids]
print(f"\nEmbedded input shape: {X.shape}")
print(f"Sample embedding (first token, first example):\n{X[0, 0, :8]}...")


# ============================================================================
# 2. POSITIONAL ENCODING
# ============================================================================
print("\n\n2. POSITIONAL ENCODING")
print("-"*60)

def get_positional_encoding(seq_len, d_model):
    """
    Sinusoidal positional encoding
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    position = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

pos_encoding = get_positional_encoding(seq_len, d_model)
print(f"Positional encoding shape: {pos_encoding.shape}")
print(f"Position 0 encoding:\n{pos_encoding[0, :8]}...")

# Add positional encoding to embeddings
X = X + pos_encoding[np.newaxis, :, :]  # Broadcasting over batch
print(f"\nInput + Positional Encoding shape: {X.shape}")


# ============================================================================
# 3. MULTI-HEAD SELF-ATTENTION
# ============================================================================
print("\n\n3. MULTI-HEAD SELF-ATTENTION")
print("-"*60)

# Weight matrices for Q, K, V projections (d_model, d_model)
W_q = np.random.randn(d_model, d_model) * 0.1
W_k = np.random.randn(d_model, d_model) * 0.1
W_v = np.random.randn(d_model, d_model) * 0.1
W_o = np.random.randn(d_model, d_model) * 0.1  # Output projection

print(f"Weight matrices: W_q, W_k, W_v, W_o all shape {W_q.shape}")

# Project to Q, K, V
Q = X @ W_q  # (batch_size, seq_len, d_model)
K = X @ W_k
V = X @ W_v

print(f"\nQ, K, V shapes: {Q.shape}")

# Reshape for multi-head attention
# (batch_size, seq_len, d_model) -> (batch_size, n_heads, seq_len, d_k)
def split_heads(x, batch_size, n_heads, seq_len, d_k):
    x = x.reshape(batch_size, seq_len, n_heads, d_k)
    return x.transpose(0, 2, 1, 3)  # (batch_size, n_heads, seq_len, d_k)

Q_heads = split_heads(Q, batch_size, n_heads, seq_len, d_k)
K_heads = split_heads(K, batch_size, n_heads, seq_len, d_k)
V_heads = split_heads(V, batch_size, n_heads, seq_len, d_k)

print(f"After splitting into heads: {Q_heads.shape}")
print(f"  (batch_size={batch_size}, n_heads={n_heads}, seq_len={seq_len}, d_k={d_k})")

# Scaled dot-product attention
# Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
scores = Q_heads @ K_heads.transpose(0, 1, 3, 2)  # (batch, n_heads, seq_len, seq_len)
scores = scores / np.sqrt(d_k)

print(f"\nAttention scores shape: {scores.shape}")
print(f"Attention scores (head 0, example 0):\n{scores[0, 0]}")

# Softmax over last dimension
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

attention_weights = softmax(scores)
print(f"\nAttention weights (head 0, example 0):\n{attention_weights[0, 0]}")
print(f"Row sums (should be 1.0): {attention_weights[0, 0].sum(axis=1)}")

# Apply attention to values
attention_output = attention_weights @ V_heads  # (batch, n_heads, seq_len, d_k)
print(f"\nAttention output shape: {attention_output.shape}")

# Concatenate heads
attention_output = attention_output.transpose(0, 2, 1, 3)  # (batch, seq_len, n_heads, d_k)
attention_output = attention_output.reshape(batch_size, seq_len, d_model)
print(f"After concatenating heads: {attention_output.shape}")

# Final linear projection
attention_output = attention_output @ W_o
print(f"After output projection: {attention_output.shape}")


# ============================================================================
# 4. ADD & NORM (Residual Connection + Layer Normalization)
# ============================================================================
print("\n\n4. ADD & NORM (after attention)")
print("-"*60)

def layer_norm(x, gamma, beta, eps=1e-6):
    """Layer normalization over the last dimension"""
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(variance + eps)
    return gamma * x_norm + beta

# Learnable parameters for layer norm
gamma_1 = np.ones(d_model)
beta_1 = np.zeros(d_model)

# Residual connection + layer norm
X_norm1 = layer_norm(X + attention_output, gamma_1, beta_1)
print(f"After residual + layer norm: {X_norm1.shape}")
print(f"Sample values (first position, first example):\n{X_norm1[0, 0, :8]}...")


# ============================================================================
# 5. FEED-FORWARD NETWORK
# ============================================================================
print("\n\n5. FEED-FORWARD NETWORK")
print("-"*60)

# FFN(x) = max(0, xW1 + b1)W2 + b2
W1 = np.random.randn(d_model, d_ff) * 0.1
b1 = np.random.randn(d_ff) * 0.1
W2 = np.random.randn(d_ff, d_model) * 0.1
b2 = np.random.randn(d_model) * 0.1

print(f"FFN weights: W1 {W1.shape}, W2 {W2.shape}")

# First linear layer + ReLU
ffn_hidden = X_norm1 @ W1 + b1  # (batch_size, seq_len, d_ff)
ffn_hidden = np.maximum(0, ffn_hidden)  # ReLU
print(f"After first FFN layer + ReLU: {ffn_hidden.shape}")

# Second linear layer
ffn_output = ffn_hidden @ W2 + b2  # (batch_size, seq_len, d_model)
print(f"After second FFN layer: {ffn_output.shape}")


# ============================================================================
# 6. ADD & NORM (after FFN)
# ============================================================================
print("\n\n6. ADD & NORM (after FFN)")
print("-"*60)

gamma_2 = np.ones(d_model)
beta_2 = np.zeros(d_model)

# Residual connection + layer norm
X_norm2 = layer_norm(X_norm1 + ffn_output, gamma_2, beta_2)
print(f"Final output shape: {X_norm2.shape}")
print(f"Sample values (first position, first example):\n{X_norm2[0, 0, :8]}...")


# ============================================================================
# 7. SUMMARY
# ============================================================================
print("\n\n" + "="*60)
print("SUMMARY OF TRANSFORMER LAYER")
print("="*60)
print(f"Input shape:              {X.shape}")
print(f"After Multi-Head Attn:    {attention_output.shape}")
print(f"After Add & Norm 1:       {X_norm1.shape}")
print(f"After Feed-Forward:       {ffn_output.shape}")
print(f"After Add & Norm 2:       {X_norm2.shape}")
print("="*60)
print("\nThis represents ONE Transformer encoder layer.")
print("A full Transformer stacks multiple such layers (e.g., 6 or 12).")
print("="*60)


# ============================================================================
# 8. BONUS: Visualize attention pattern
# ============================================================================
print("\n\n8. ATTENTION PATTERN VISUALIZATION")
print("-"*60)
print("Attention weights for first example, head 0:")
print("(rows = query positions, cols = key positions)")
print()
print("     Pos0   Pos1   Pos2   Pos3")
for i in range(seq_len):
    print(f"Pos{i}", end="  ")
    for j in range(seq_len):
        print(f"{attention_weights[0, 0, i, j]:.3f}", end="  ")
    print()
print("\nEach row sums to 1.0 (probability distribution)")

