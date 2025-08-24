# Transformers: Attention is All You Need

## Learning Objectives
- Understand the transformer architecture and its key innovations
- Learn how self-attention mechanisms work
- Understand the role of positional encoding
- Learn about encoder-decoder architectures vs encoder-only models
- Understand how transformers revolutionized NLP and beyond
- Learn about modern transformer variants (BERT, GPT, T5, etc.)

## Outline (15-minute lecture)

### 1. Introduction to Transformers
- The revolution in sequence modeling
- Problems with RNNs and CNNs for sequences
- "Attention is All You Need" (Vaswani et al., 2017)
- Key innovation: parallelizable sequence processing

### 2. The Attention Mechanism
- Motivation: why attention matters
- Query, Key, Value paradigm
- Scaled dot-product attention
- Mathematical formulation

### 3. Self-Attention
- Self-attention vs cross-attention
- How tokens attend to each other
- Attention weights and their interpretation
- Computational complexity: O(n²)

### 4. Multi-Head Attention
- Multiple attention "heads" in parallel
- Different representation subspaces
- Concatenation and linear projection
- Why multiple heads help capture different relationships

### 5. The Transformer Block
- Layer normalization and residual connections
- Position-wise feed-forward networks
- Complete transformer layer architecture
- Stacking multiple layers

### 6. Positional Encoding
- The problem: no inherent position information
- Sinusoidal positional encoding
- Learned vs fixed positional embeddings
- Relative position representations

### 7. Encoder-Decoder Architecture
- Original transformer design for translation
- Encoder stack processing input sequence
- Decoder stack with masked self-attention
- Cross-attention between encoder and decoder

### 8. Modern Transformer Variants
- BERT: Bidirectional encoder representations
- GPT: Generative pre-trained transformers
- T5: Text-to-text transfer transformer
- Vision transformers and multimodal models

### 9. Training and Optimization
- Pre-training objectives (MLM, causal LM)
- Fine-tuning for downstream tasks
- Computational requirements and scaling
- Optimization challenges and solutions

### 10. Key Takeaways
- Attention enables parallelizable sequence modeling
- Self-attention captures long-range dependencies
- Transformer architecture is highly modular and scalable
- Foundation for modern large language models
- Applications beyond NLP: vision, multimodal AI
