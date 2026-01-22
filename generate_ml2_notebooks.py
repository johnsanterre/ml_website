#!/usr/bin/env python3
"""
Generate Jupyter notebooks for ML2 course lectures.
Each notebook contains teaching material and 10 questions.
"""

import json
import os

# ML2 Lecture Topics and Content
LECTURES = {
    1: {
        "title": "Introduction to Deep Learning",
        "topics": ["Neural network basics", "Activation functions", "Loss functions", "Forward propagation"],
        "concepts": [
            "What is deep learning and how does it differ from traditional ML?",
            "Understanding neurons and layers in neural networks",
            "Common activation functions: ReLU, Sigmoid, Tanh",
            "Introduction to gradient descent and backpropagation"
        ]
    },
    2: {
        "title": "Neural Networks & Backpropagation",
        "topics": ["Backpropagation", "Gradient descent variants", "Loss functions", "Training challenges"],
        "concepts": [
            "Chain rule and backpropagation algorithm",
            "Batch vs Mini-batch vs Stochastic Gradient Descent",
            "Vanishing and exploding gradients",
            "Cross-entropy vs MSE loss functions"
        ]
    },
    3: {
        "title": "Building Real-World Housing Price Predictor",
        "topics": ["Data preprocessing", "Feature engineering", "Model training", "Evaluation metrics"],
        "concepts": [
            "Loading and exploring the California Housing dataset",
            "Feature normalization and standardization",
            "Building a neural network regressor in PyTorch",
            "Evaluating regression models: MSE, RMSE, R²"
        ]
    },
    4: {
        "title": "Vector Representations & Similarity Measures",
        "topics": ["Word embeddings", "Similarity metrics", "Dimensionality reduction", "Sparse vs dense representations"],
        "concepts": [
            "One-hot encoding vs learned embeddings",
            "Cosine similarity and Euclidean distance",
            "Understanding embedding spaces",
            "Applications: recommendation systems, semantic search"
        ]
    },
    5: {
        "title": "Autoencoders & Embeddings",
        "topics": ["Autoencoder architecture", "Latent space", "Dimensionality reduction", "Denoising autoencoders"],
        "concepts": [
            "Encoder-decoder architecture",
            "Learning compressed representations",
            "Reconstruction loss and regularization",
            "Applications: anomaly detection, image denoising"
        ]
    },
    6: {
        "title": "From Autoencoders to Embeddings",
        "topics": ["Word2Vec", "GloVe", "Skip-gram", "CBOW", "Embedding applications"],
        "concepts": [
            "Word2Vec: Skip-gram vs CBOW models",
            "GloVe: Global Vectors for word representation",
            "Embedding arithmetic and analogies",
            "Using pre-trained embeddings in downstream tasks"
        ]
    },
    7: {
        "title": "Sequence Models & Attention",
        "topics": ["RNNs", "LSTMs", "GRUs", "Attention mechanism", "Sequence-to-sequence"],
        "concepts": [
            "Recurrent Neural Networks and sequential data",
            "LSTM cells: forget, input, output gates",
            "Attention mechanism fundamentals",
            "Applications: machine translation, time series"
        ]
    },
    8: {
        "title": "Convolutional Neural Networks",
        "topics": ["Convolution operation", "Pooling", "CNN architectures", "Transfer learning"],
        "concepts": [
            "Convolution filters and feature maps",
            "MaxPooling and spatial downsampling",
            "Classic architectures: LeNet, AlexNet, VGG, ResNet",
            "Transfer learning and fine-tuning"
        ]
    },
    9: {
        "title": "From Supervised to Generative Learning",
        "topics": ["Generative models", "VAEs", "GANs", "Latent space manipulation"],
        "concepts": [
            "Discriminative vs Generative models",
            "Variational Autoencoders (VAEs)",
            "Generative Adversarial Networks (GANs)",
            "Applications: image generation, style transfer"
        ]
    },
    10: {
        "title": "Introduction to Large Language Models",
        "topics": ["Transformer architecture", "Self-attention", "Tokenization", "Pre-training"],
        "concepts": [
            "Transformer architecture: encoder-decoder",
            "Multi-head self-attention mechanism",
            "Positional encoding in transformers",
            "Pre-training objectives: MLM, causal LM"
        ]
    },
    11: {
        "title": "Practical LLM Integration & API Development",
        "topics": ["API usage", "Prompt engineering", "Token management", "Cost optimization"],
        "concepts": [
            "Using OpenAI/Anthropic APIs effectively",
            "Prompt engineering best practices",
            "Managing context windows and tokens",
            "Rate limiting and error handling"
        ]
    },
    12: {
        "title": "Retrieval Augmented Generation (RAG)",
        "topics": ["Vector databases", "Semantic search", "RAG pipeline", "Chunking strategies"],
        "concepts": [
            "RAG architecture: retrieval + generation",
            "Vector databases: FAISS, Pinecone, Weaviate",
            "Document chunking and embedding",
            "Improving retrieval quality"
        ]
    },
    13: {
        "title": "Evaluating LLMs - Metrics and Methods",
        "topics": ["Evaluation metrics", "Benchmarks", "Human evaluation", "Automated scoring"],
        "concepts": [
            "Perplexity and cross-entropy for LMs",
            "BLEU, ROUGE, METEOR for generation",
            "Human evaluation: relevance, coherence, factuality",
            "Benchmark datasets: GLUE, SuperGLUE, MMLU"
        ]
    },
    14: {
        "title": "LLMs as Decision Makers and Agents",
        "topics": ["Agent frameworks", "Tool use", "Planning", "Multi-step reasoning"],
        "concepts": [
            "LLM agents and autonomous systems",
            "Tool calling and function execution",
            "ReAct: Reasoning + Acting framework",
            "Multi-agent systems and collaboration"
        ]
    },
    15: {
        "title": "Future Trends in LLMs",
        "topics": ["Multimodal models", "Efficient training", "Alignment", "Emerging capabilities"],
        "concepts": [
            "Multimodal LLMs: vision + language",
            "Efficient fine-tuning: LoRA, QLoRA, PEFT",
            "Alignment and RLHF",
            "Emerging capabilities and scaling laws"
        ]
    }
}

def create_notebook(lecture_num, lecture_info):
    """Create a Jupyter notebook for a given lecture."""
    
    cells = []
    
    # Title cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# Week {lecture_num}: {lecture_info['title']}\\n",
            "\\n",
            "**ML2: Advanced Machine Learning**\\n",
            "\\n",
            "---\\n",
            "\\n",
            "This notebook contains teaching material, examples, and 10 practice questions to reinforce your understanding of this week's topics."
        ]
    })
    
    # Learning Objectives
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Learning Objectives\\n",
            "\\n",
            "By the end of this week, you should be able to:\\n",
            "\\n"
        ] + [f"- {concept}\\n" for concept in lecture_info['concepts']]
    })
    
    # Prerequisites / Setup
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Setup\\n",
            "\\n",
            "Run this cell to import necessary libraries:"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import numpy as np\\n",
            "import matplotlib.pyplot as plt\\n",
            "import torch\\n",
            "import torch.nn as nn\\n",
            "import torch.optim as optim\\n",
            "\\n",
            "# Set random seeds for reproducibility\\n",
            "np.random.seed(42)\\n",
            "torch.manual_seed(42)\\n",
            "\\n",
            "print('✓ Libraries imported successfully')"
        ]
    })
    
    # Teaching Material Section
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\\n",
            "## Teaching Material\\n",
            "\\n",
            f"### Key Topics for Week {lecture_num}\\n",
            "\\n"
        ] + [f"{i+1}. **{topic}**\\n" for i, topic in enumerate(lecture_info['topics'])]
    })
    
    # Add a brief example/demo (placeholder for now)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Example: Core Concept Demonstration\\n",
            "\\n",
            "Below is a simple example demonstrating one of this week's key concepts:"
        ]
    })
    
    title_str = lecture_info["title"]
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Example code demonstrating a key concept\\n",
            "# TODO: Add specific examples for each lecture\\n",
            "\\n",
            f"print('Example for Week {lecture_num}: {title_str}')"
        ]
    })
    
    # Practice Questions Section
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\\n",
            "## Practice Questions\\n",
            "\\n",
            "Complete the following 10 questions to test your understanding. Questions progress from conceptual to applied.\\n",
            "\\n",
            "**Instructions:**\\n",
            "- Read each question carefully\\n",
            "- Fill in code cells where indicated\\n",
            "- Check your understanding with the provided hints\\n",
            "- Solutions are available in a separate file"
        ]
    })
    
    # Generate 10 questions (mix of markdown explanations + code cells)
    for q_num in range(1, 11):
        # Question cell
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"### Question {q_num}\\n",
                "\\n",
                f"**Topic:** {lecture_info['topics'][(q_num-1) % len(lecture_info['topics'])]}\\n",
                "\\n",
                f"[Question text for Question {q_num} - to be customized per lecture]\\n",
                "\\n",
                "**Hint:** Think about how this concept relates to the learning objectives above."
            ]
        })
        
        # Answer cell
        if q_num % 3 == 0:  # Every 3rd question is conceptual (markdown)
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "**Your Answer:**\\n",
                    "\\n",
                    "[Write your answer here]"
                ]
            })
        else:  # Coding questions
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Your code here\\n",
                    "\\n"
                ]
            })
    
    # Reflection section
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\\n",
            "## Reflection\\n",
            "\\n",
            "Take a moment to reflect on what you've learned:\\n",
            "\\n",
            "1. What was the most challenging concept this week?\\n",
            "2. How might you apply these concepts to a real-world problem?\\n",
            "3. What questions do you still have?\\n",
            "\\n",
            "**Your reflections:**"
        ]
    })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "[Write your reflections here]"
        ]
    })
    
    # Additional Resources
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\\n",
            "## Additional Resources\\n",
            "\\n",
            "- Course textbook: See week's reading assignments\\n",
            "- Office hours: Check course schedule\\n",
            f"- Course website: [Week {lecture_num} Materials](../../ml2/lecture{lecture_num:02d}/index.html)\\n",
            "\\n",
            "**Next Steps:**\\n",
            "- Review any questions you found challenging\\n",
            "- Complete the week's coding assignments\\n",
            "- Prepare for next week's topics"
        ]
    })
    
    # Create notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook

def main():
    """Generate all ML2 notebooks."""
    
    for lecture_num in range(1, 16):
        lecture_info = LECTURES[lecture_num]
        notebook = create_notebook(lecture_num, lecture_info)
        
        # Create output directory if it doesn't exist
        output_dir = f"ml2/lecture{lecture_num:02d}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Write notebook file
        output_path = f"{output_dir}/week_{lecture_num:02d}_exercises.ipynb"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Created {output_path}")
    
    print(f"\\n🎉 Successfully generated {len(LECTURES)} notebooks!")

if __name__ == "__main__":
    main()
