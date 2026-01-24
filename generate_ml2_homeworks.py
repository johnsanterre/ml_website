#!/usr/bin/env python3
"""
Generate homework assignments for ML2 course lectures.
Each homework combines programming exercises and knowledge questions.
Target completion time: ~1 hour
"""

import json
import os

# Homework content for each lecture
HOMEWORKS = {
    1: {
        "title": "Introduction to Deep Learning",
        "programming": [
            {
                "title": "Question 1: Implement ReLU",
                "description": "Implement the ReLU (Rectified Linear Unit) activation function from scratch.",
                "time": "3 min",
                "starter_code": """import numpy as np

def relu(x):
    # TODO: Implement ReLU - returns max(0, x)
    pass

# Test
test_input = np.array([-2, -1, 0, 1, 2])
print(relu(test_input))  # Should output: [0, 0, 0, 1, 2]"""
            },
            {
                "title": "Question 2: Implement Sigmoid",
                "description": "Implement the Sigmoid activation function.",
                "time": "4 min",
                "starter_code": """import numpy as np

def sigmoid(x):
    # TODO: Implement Sigmoid - returns 1 / (1 + e^(-x))
    pass

# Test
test_input = np.array([-2, 0, 2])
print(sigmoid(test_input))  # Should be close to [0.12, 0.5, 0.88]"""
            },
            {
                "title": "Question 3: Create a Linear Layer",
                "description": "Create a simple linear (fully-connected) layer using PyTorch.",
                "time": "5 min",
                "starter_code": """import torch
import torch.nn as nn

# TODO: Create a linear layer that transforms 10 inputs to 5 outputs
linear_layer = None

# Test
x = torch.randn(1, 10)  # batch_size=1, input_size=10
output = linear_layer(x)
print(output.shape)  # Should be torch.Size([1, 5])"""
            },
            {
                "title": "Question 4: Forward Pass Calculation",
                "description": "Manually compute the output of a simple network.",
                "time": "6 min",
                "starter_code": """import torch

# Given weights and input
x = torch.tensor([1.0, 2.0])
W = torch.tensor([[0.5, 0.3], [0.2, 0.6]])
b = torch.tensor([0.1, 0.1])

# TODO: Compute y = W @ x + b (matrix multiplication)
y = None

print(y)  # Verify your calculation"""
            },
            {
                "title": "Question 5: Apply Activation to Layer",
                "description": "Chain a linear layer with a ReLU activation.",
                "time": "5 min",
                "starter_code": """import torch
import torch.nn as nn

# TODO: Create a Sequential model with:
# - Linear layer (input=20, output=10)
# - ReLU activation
model = None

# Test
x = torch.randn(5, 20)  # batch of 5 samples
output = model(x)
print(output.shape)  # Should be torch.Size([5, 10])"""
            },
            {
                "title": "Question 6: Count Parameters",
                "description": "Calculate the number of parameters in a neural network layer.",
                "time": "4 min",
                "starter_code": """import torch.nn as nn

# Given layer
layer = nn.Linear(100, 50)

# TODO: Calculate total number of parameters (weights + biases)
# Hint: Weights are 100 x 50, biases are 50
num_params = None

print(f\"Total parameters: {num_params}\")  # Should be 5050"""
            }
        ],
        "knowledge": [
            {
                "type": "mc",
                "question": "Question 7: What is the main advantage of deep learning over traditional machine learning?",
                "options": [
                    "A) Requires less data",
                    "B) Automatically learns hierarchical features",
                    "C) Always runs faster",
                    "D) Doesn't need any hyperparameter tuning"
                ],
                "hint": "Think about feature engineering in traditional ML vs deep learning."
            },
            {
                "type": "mc",
                "question": "Question 8: Which activation function outputs values in the range (0, 1)?",
                "options": [
                    "A) ReLU",
                    "B) Tanh",
                    "C) Sigmoid",
                    "D) Linear"
                ],
                "hint": "Consider which activation is commonly used for binary classification."
            },
            {
                "type": "short",
                "question": "Question 9: In 1-2 sentences, explain what a 'bias' term does in a neural network layer.",
                "hint": "Think about what happens when all inputs are zero."
            },
            {
                "type": "code_reading",
                "question": "Question 10: What will be the output shape given input shape (32, 784)?",
                "code": """nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)""",
                "hint": "Trace the dimensions through each layer. Batch size stays the same."
            }
        ]
    },
    2: {
        "title": "Neural Networks & Backpropagation",
        "programming": [
            {
                "title": "Compute Gradients Manually",
                "description": "Calculate gradients for a simple network using chain rule, then verify with PyTorch autograd.",
                "time": "20 min",
                "starter_code": """import torch

# Create simple computation graph
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# Forward pass: y = w*x + b, loss = (y - target)^2
target = torch.tensor(10.0)

# TODO: Compute forward pass and loss
# TODO: Manually calculate gradients using chain rule
# TODO: Verify with PyTorch autograd"""
            },
            {
                "title": "Implement Gradient Descent Variants",
                "description": "Implement and compare SGD, Mini-batch GD, and Batch GD on a toy dataset.",
                "time": "20 min",
                "starter_code": """import numpy as np

def batch_gd(X, y, lr=0.01, epochs=100):
    # TODO: Implement batch gradient descent
    pass

def mini_batch_gd(X, y, batch_size=32, lr=0.01, epochs=100):
    # TODO: Implement mini-batch gradient descent
    pass

# Generate toy data and compare methods"""
            }
        ],
        "knowledge": [
            {
                "type": "mc",
                "question": "Which statement about batch size is TRUE?",
                "options": [
                    "A) Larger batch sizes always lead to better generalization",
                    "B) Smaller batch sizes provide more frequent updates but noisier gradients",
                    "C) Batch size has no effect on training dynamics",
                    "D) Batch size must always equal the dataset size"
                ],
                "hint": "Consider the trade-off between update frequency and gradient quality."
            },
            {
                "type": "short",
                "question": "Why is the choice of learning rate critical? What happens if it's too large or too small?",
                "hint": "Think about convergence speed and stability."
            }
        ]
    },
    3: {
        "title": "Building Real-World Housing Price Predictor",
        "programming": [
            {
                "title": "Data Preprocessing Pipeline",
                "description": "Load California Housing dataset, handle missing values, and normalize features.",
                "time": "15 min",
                "starter_code": """from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np

# TODO: Load dataset
# TODO: Check for missing values
# TODO: Normalize features using StandardScaler
# TODO: Split into train/test sets"""
            },
            {
                "title": "Train and Evaluate Model",
                "description": "Build a neural network for regression and evaluate using appropriate metrics.",
                "time": "25 min",
                "starter_code": """import torch
import torch.nn as nn

# TODO: Define regression model
# TODO: Set up loss function (MSE) and optimizer
# TODO: Training loop
# TODO: Evaluate on test set using MSE, RMSE, and R²"""
            }
        ],
        "knowledge": [
            {
                "type": "mc",
                "question": "For regression tasks, which loss function is most commonly used?",
                "options": [
                    "A) Cross-entropy loss",
                    "B) Mean Squared Error (MSE)",
                    "C) Hinge loss",
                    "D) KL divergence"
                ],
                "hint": "Think about what we're trying to minimize in regression."
            },
            {
                "type": "short",
                "question": "Why is feature normalization important for neural networks? What could happen without it?",
                "hint": "Consider the scale of gradients and convergence."
            }
        ]
    },
    4: {
        "title": "Vector Representations & Similarity",
        "programming": [
            {
                "title": "Implement Similarity Metrics",
                "description": "Code cosine similarity and Euclidean distance from scratch.",
                "time": "15 min",
                "starter_code": """import numpy as np

def cosine_similarity(v1, v2):
    # TODO: Implement cosine similarity
    pass

def euclidean_distance(v1, v2):
    # TODO: Implement Euclidean distance
    pass

# Test with example vectors"""
            },
            {
                "title": "Build Simple Recommendation System",
                "description": "Use embeddings to find similar items based on user preferences.",
                "time": "25 min",
                "starter_code": """import numpy as np

# Sample item embeddings (e.g., movies)
item_embeddings = np.random.randn(100, 50)  # 100 items, 50-dim embeddings

def find_similar_items(item_id, embeddings, top_k=5):
    # TODO: Compute similarity between item and all others
    # TODO: Return top-k most similar items
    pass"""
            }
        ],
        "knowledge": [
            {
                "type": "mc",
                "question": "What does cosine similarity measure?",
                "options": [
                    "A) The distance between two vectors",
                    "B) The angle between two vectors",
                    "C) The magnitude of vectors",
                    "D) The dimension of vectors"
                ],
                "hint": "Cosine similarity ranges from -1 to 1, what does this represent?"
            },
            {
                "type": "short",
                "question": "Explain the curse of dimensionality and how it affects similarity search in high dimensions.",
                "hint": "What happens to distances as dimensionality increases?"
            }
        ]
    },
    5: {
        "title": "Autoencoders & Embeddings",
        "programming": [
            {
                "title": "Build a Simple Autoencoder",
                "description": "Implement an autoencoder for MNIST digit compression.",
                "time": "20 min",
                "starter_code": """import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32):
        super(Autoencoder, self).__init__()
        # TODO: Define encoder
        # TODO: Define decoder
        pass
    
    def forward(self, x):
        # TODO: Encode and decode
        pass

# TODO: Train on MNIST"""
            },
            {
                "title": "Visualize Latent Space",
                "description": "Extract and visualize the learned latent representations.",
                "time": "20 min",
                "starter_code": """import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# TODO: Encode test images to latent space
# TODO: Use t-SNE to reduce to 2D
# TODO: Plot colored by digit class"""
            }
        ],
        "knowledge": [
            {
                "type": "mc",
                "question": "What is the main purpose of the bottleneck layer in an autoencoder?",
                "options": [
                    "A) To slow down training",
                    "B) To force compression and learn important features",
                    "C) To increase model capacity",
                    "D) To prevent overfitting only"
                ],
                "hint": "Why do we make the latent dimension smaller than the input?"
            },
            {
                "type": "short",
                "question": "How do Variational Autoencoders (VAEs) differ from standard autoencoders? What additional capability do they have?",
                "hint": "Think about Point estimates vs distributions."
            }
        ]
    },
    # Continuing with lectures 6-15...
    6: {
        "title": "From Autoencoders to Embeddings",
        "programming": [
            {
                "title": "Word Embedding Analogies",
                "description": "Use pre-trained word embeddings to solve analogy tasks.",
                "time": "15 min",
                "starter_code": """# Using pre-trained embeddings (e.g., Word2Vec or GloVe)
# TODO: Load embeddings
# TODO: Implement function to solve analogies: king - man + woman = ?
# TODO: Test with various analogies"""
            },
            {
                "title": "Visualize Word Embeddings",
                "description": "Create a 2D visualization of word embedding space.",
                "time": "25 min",
                "starter_code": """from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# TODO: Select subset of words
# TODO: Reduce to 2D using PCA or t-SNE
# TODO: Plot and label points"""
            }
        ],
        "knowledge": [
            {
                "type": "mc",
                "question": "What is the main difference between Word2Vec and GloVe?",
                "options": [
                    "A) Word2Vec is supervised, GloVe is unsupervised",
                    "B) Word2Vec captures local context, GloVe uses global co-occurrence statistics",
                    "C) GloVe only works for English",
                    "D) Word2Vec cannot handle rare words"
                ],
                "hint": "Consider the training objectives of each method."
            },
            {
                "type": "short",
                "question": "Explain how negative sampling helps make Word2Vec training efficient.",
                "hint": "What's the problem with computing softmax over the entire vocabulary?"
            }
        ]
    },
    7: {
        "title": "Sequence Models & Attention",
        "programming": [
            {
                "title": "Implement Simple RNN",
                "description": "Build a basic RNN cell from scratch.",
                "time": "20 min",
                "starter_code": """import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        # TODO: Define parameters
        pass
    
    def forward(self, x, hidden=None):
        # TODO: Implement RNN cell
        pass"""
            },
            {
                "title": "Attention Mechanism",
                "description": "Implement scaled dot-product attention.",
                "time": "20 min",
                "starter_code": """import torch
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value):
    # TODO: Implement attention
    # scores = QK^T / sqrt(d_k)
    # weights = softmax(scores)
    # output = weights * V
    pass"""
            }
        ],
        "knowledge": [
            {
                "type": "mc",
                "question": "Why do LSTMs help with the vanishing gradient problem compared to vanilla RNNs?",
                "options": [
                    "A) They have more parameters",
                    "B) They use gates to control information flow",
                    "C) They train faster",
                    "D) They don't use backpropagation"
                ],
                "hint": "Think about the cell state and gate mechanisms."
            },
            {
                "type": "short",
                "question": "Explain the key innovation of the attention mechanism. Why is it better than using only the last hidden state?",
                "hint": "Consider long sequences and information bottleneck."
            }
        ]
    },
    8: {
        "title": "Convolutional Neural Networks",
        "programming": [
            {
                "title": "Build CNN for Image Classification",
                "description": "Create a CNN for CIFAR-10 classification.",
                "time": "25 min",
                "starter_code": """import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        # TODO: Define convolutional layers
        # TODO: Define pooling layers
        # TODO: Define fully connected layers
        pass
    
    def forward(self, x):
        # TODO: Implement forward pass
        pass"""
            },
            {
                "title": "Visualize Feature Maps",
                "description": "Extract and visualize intermediate feature maps from your CNN.",
                "time": "15 min",
                "starter_code": """# TODO: Register hooks to extract activations
# TODO: Pass image through network
# TODO: Visualize feature maps from different layers"""
            }
        ],
        "knowledge": [
            {
                "type": "mc",
                "question": "What is the main advantage of convolutional layers over fully connected layers for images?",
                "options": [
                    "A) Fewer parameters due to weight sharing",
                    "B) They train faster",
                    "C) They don't need activation functions",
                    "D) They work only with color images"
                ],
                "hint": "Think about spatial structure and parameter efficiency."
            },
            {
                "type": "short",
                "question": "Explain the purpose of pooling layers in CNNs. What are the trade-offs?",
                "hint": "Consider spatial dimensions, translation invariance, and information loss."
            }
        ]
    },
    9: {
        "title": "From Supervised to Generative Learning",
        "programming": [
            {
                "title": "Implement VAE Loss",
                "description": "Code the VAE loss function combining reconstruction and KL divergence.",
                "time": "20 min",
                "starter_code": """import torch
import torch.nn.functional as F

def vae_loss(recon_x, x, mu, logvar):
    # TODO: Compute reconstruction loss
    # TODO: Compute KL divergence
    # TODO: Combine and return total loss
    pass"""
            },
            {
                "title": "Generate New Samples",
                "description": "Use trained VAE decoder to generate novel images.",
                "time": "20 min",
                "starter_code": """# TODO: Sample from latent space (normal distribution)
# TODO: Pass through decoder
# TODO: Visualize generated images"""
            }
        ],
        "knowledge": [
            {
                "type": "mc",
                "question": "What is the key difference between discriminative and generative models?",
                "options": [
                    "A) Generative models only work with images",
                    "B) Discriminative models learn P(y|x), generative models learn P(x,y) or P(x)",
                    "C) Generative models are always faster",
                    "D) Discriminative models cannot use deep learning"
                ],
                "hint": "Think about what each type of model is learning."
            },
            {
                "type": "short",
                "question": "Explain the reparameterization trick in VAEs and why it's necessary.",
                "hint": "How do we backpropagate through a sampling operation?"
            }
        ]
    },
    10: {
        "title": "Introduction to Large Language Models",
        "programming": [
            {
                "title": "Use OpenAI API",
                "description": "Make basic API calls to GPT and experiment with parameters.",
                "time": "20 min",
                "starter_code": """from openai import OpenAI
client = OpenAI()

# TODO: Make completion request
# TODO: Experiment with temperature and top_p
# TODO: Compare outputs with different parameters"""
            },
            {
                "title": "Build Simple Chatbot",
                "description": "Create a conversational interface using the API.",
                "time": "20 min",
                "starter_code": """# TODO: Maintain conversation history
# TODO: Add system message
# TODO: Handle multi-turn dialogue"""
            }
        ],
        "knowledge": [
            {
                "type": "mc",
                "question": "What does the temperature parameter control in LLM sampling?",
                "options": [
                    "A) The speed of generation",
                    "B) The randomness/diversity of outputs",
                    "C) The model size",
                    "D) The number of tokens generated"
                ],
                "hint": "Higher temperature = more random outputs."
            },
            {
                "type": "short",
                "question": "Explain what the transformer architecture's self-attention mechanism computes.",
                "hint": "How does each token attend to other tokens?"
            }
        ]
    },
    11: {
        "title": "Practical LLM Integration",
        "programming": [
            {
                "title": "Prompt Engineering",
                "description": "Design and test effective prompts for specific tasks.",
                "time": "20 min",
                "starter_code": """# TODO: Create prompts for:
# - Sentiment analysis
# - Information extraction
# - Text summarization
# Test and compare effectiveness"""
            },
            {
                "title": "Function Calling",
                "description": "Use LLM function calling to integrate external tools.",
                "time": "20 min",
                "starter_code": """# TODO: Define function schemas
# TODO: Make API call with functions
# TODO: Parse and execute function calls"""
            }
        ],
        "knowledge": [
            {
                "type": "mc",
                "question": "What is few-shot prompting?",
                "options": [
                    "A) Using multiple prompts at once",
                    "B) Providing examples in the prompt to guide the model",
                    "C) Training the model on few examples",
                    "D) Making fast API calls"
                ],
                "hint": "Think about in-context learning."
            },
            {
                "type": "short",
                "question": "Why is token counting important when using LLM APIs? What are the implications for cost and performance?",
                "hint": "Consider pricing models and context window limits."
            }
        ]
    },
    12: {
        "title": "Retrieval Augmented Generation",
        "programming": [
            {
                "title": "Build Simple RAG System",
                "description": "Implement basic RAG pipeline with embeddings and retrieval.",
                "time": "30 min",
                "starter_code": """# TODO: Load documents
# TODO: Create embeddings
# TODO: Store in vector database (or simple list)
# TODO: Implement retrieval function
# TODO: Augment prompt with retrieved context"""
            },
            {
                "title": "Evaluate Retrieval Quality",
                "description": "Test retrieval with various queries and measure relevance.",
                "time": "10 min",
                "starter_code": """# TODO: Create test queries
# TODO: Retrieve top-k documents
# TODO: Manually assess relevance"""
            }
        ],
        "knowledge": [
            {
                "type": "mc",
                "question": "What problem does RAG primarily solve?",
                "options": [
                    "A) Making models smaller",
                    "B) Providing models with up-to-date and specific external knowledge",
                    "C) Speeding up inference",
                    "D) Eliminating need for training"
                ],
                "hint": "Think about limitations of parametric knowledge."
            },
            {
                "type": "short",
                "question": "Explain the trade-off between chunk size in document splitting for RAG.",
                "hint": "Consider precision vs context completeness."
            }
        ]
    },
    13: {
        "title": "Evaluating LLMs",
        "programming": [
            {
                "title": "Implement Evaluation Metrics",
                "description": "Code BLEU, ROUGE, and perplexity from scratch.",
                "time": "25 min",
                "starter_code": """def bleu_score(reference, candidate):
    # TODO: Implement BLEU
    pass

def rouge_score(reference, candidate):
    # TODO: Implement ROUGE
    pass

def perplexity(model, text):
    # TODO: Compute perplexity
    pass"""
            },
            {
                "title": "Create Evaluation Suite",
                "description": "Build test cases and run systematic evaluation.",
                "time": "15 min",
                "starter_code": """# TODO: Define test cases
# TODO: Run model on tests
# TODO: Compute metrics
# TODO: Generate evaluation report"""
            }
        ],
        "knowledge": [
            {
                "type": "mc",
                "question": "What does perplexity measure?",
                "options": [
                    "A) Model size",
                    "B) How surprised the model is by the text",
                    "C) Training time",
                    "D) Number of parameters"
                ],
                "hint": "Lower perplexity means the model predicts the text better."
            },
            {
                "type": "short",
                "question": "Why are automatic metrics like BLEU insufficient for evaluating LLMs? What's missing?",
                "hint": "Consider the diversity of valid responses."
            }
        ]
    },
    14: {
        "title": "LLMs as Decision Makers",
        "programming": [
            {
                "title": "Build ReAct Agent",
                "description": "Implement simple reasoning and acting loop.",
                "time": "25 min",
                "starter_code": """# TODO: Define tools (functions the agent can call)
# TODO: Create agent loop:
#   - Think (reason about what to do)
#   - Act (call tool)
#   - Observe (get results)
# TODO: Test with multi-step task"""
            },
            {
                "title": "Tool Use Evaluation",
                "description": "Test agent's ability to choose and use appropriate tools.",
                "time": "15 min",
                "starter_code": """# TODO: Create test scenarios
# TODO: Track tool selection accuracy
# TODO: Measure task completion rate"""
            }
        ],
        "knowledge": [
            {
                "type": "mc",
                "question": "What is the main challenge in using LLMs as agents?",
                "options": [
                    "A) They're too slow",
                    "B) Ensuring reliable and safe decision making",
                    "C) They can't use tools",
                    "D) They only work with text"
                ],
                "hint": "Consider errors, hallucinations, and consequences of actions."
            },
            {
                "type": "short",
                "question": "Explain the ReAct framework. How does it combine reasoning and acting?",
                "hint": "Think about the thought-action-observation loop."
            }
        ]
    },
    15: {
        "title": "Future Trends in LLMs",
        "programming": [
            {
                "title": "Experiment with Multimodal Model",
                "description": "Use GPT-4V or similar to analyze images and text together.",
                "time": "20 min",
                "starter_code": """# TODO: Load image and create prompt
# TODO: Send to multimodal API
# TODO: Test various vision-language tasks"""
            },
            {
                "title": "Compare Model Sizes",
                "description": "Benchmark different model sizes on the same task.",
                "time": "20 min",
                "starter_code": """# TODO: Test same prompts on different models
# TODO: Compare: quality, latency, cost
# TODO: Analyze scaling behavior"""
            }
        ],
        "knowledge": [
            {
                "type": "mc",
                "question": "What is a key benefit of LoRA (Low-Rank Adaptation)?",
                "options": [
                    "A) Makes models larger",
                    "B) Enables efficient fine-tuning with few parameters",
                    "C) Only works for vision models",
                    "D) Eliminates need for GPUs"
                ],
                "hint": "Think about parameter-efficient fine-tuning."
            },
            {
                "type": "short",
                "question": "Explain the alignment problem in LLMs. Why is RLHF necessary?",
                "hint": "What's the difference between predicting next tokens and being helpful/safe?"
            }
        ]
    }
}

def create_homework_notebook(hw_num, hw_info):
    """Create homework notebook for a given lecture."""
    cells = []
    
    # Title and overview
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# Week {hw_num}: {hw_info['title']} - Homework\n",
            "\n",
            "**ML2: Advanced Machine Learning**\n",
            "\n",
            "**Estimated Time**: 1 hour\n",
            "\n",
            "---\n",
            "\n",
            "This homework combines programming exercises and knowledge-based questions to reinforce this week's concepts."
        ]
    })
    
    # Setup
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Setup\n",
            "\n",
            "Run this cell to import necessary libraries:"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "\n",
            "# Set random seed for reproducibility\n",
            "np.random.seed(42)\n",
            "torch.manual_seed(42)\n",
            "\n",
            "print('✓ Libraries imported successfully')"
        ]
    })
    
    # Programming Exercises
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## Part 1: Programming Exercises (60%)\n",
            "\n",
            "Complete the following programming tasks. Read each description carefully and implement the requested functionality."
        ]
    })
    
    for i, exercise in enumerate(hw_info['programming'], 1):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"### Exercise {i}: {exercise['title']}\n",
                "\n",
                f"**Time**: {exercise['time']}\n",
                "\n",
                f"{exercise['description']}"
            ]
        })
        
        # Split code and add newlines back to each line
        code_lines = exercise['starter_code'].split('\n')
        code_source = [line + '\n' for line in code_lines[:-1]] + ([code_lines[-1]] if code_lines[-1] else [])
        
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code_source
        })
    
    # Knowledge Questions
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## Part 2: Knowledge Questions (40%)\n",
            "\n",
            "Answer the following questions to test your conceptual understanding."
        ]
    })
    
    for i, question in enumerate(hw_info['knowledge'], 1):
        if question['type'] == 'mc':
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"### Question {i} (Multiple Choice)\n",
                    "\n",
                    f"{question['question']}\n",
                    "\n"
                ] + [f"{opt}\n" for opt in question['options']] + [
                    "\n",
                    f"**Hint**: {question['hint']}"
                ]
            })
            
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "**Your Answer**: [Write your answer here - e.g., 'B']\n",
                    "\n",
                    "**Explanation**: [Explain why this is correct]"
                ]
            })
        
        elif question['type'] == 'short':
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"### Question {i} (Short Answer)\n",
                    "\n",
                    f"{question['question']}\n",
                    "\n",
                    f"**Hint**: {question['hint']}"
                ]
            })
            
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "**Your Answer**:\n",
                    "\n",
                    "[Write your answer here in 2-4 sentences]"
                ]
            })
        
        elif question['type'] == 'code_reading':
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"### Question {i} (Code Reading)\n",
                    "\n",
                    f"{question['question']}\n",
                    "\n",
                    "```python\n",
                    f"{question['code']}\n",
                    "```\n",
                    "\n",
                    f"**Hint**: {question['hint']}"
                ]
            })
            
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "**Your Answer**: [Write your answer here]"
                ]
            })
    
    # Submission
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## Submission\n",
            "\n",
            "Before submitting:\n",
            "1. Run all cells to ensure code executes without errors\n",
            "2. Check that all questions are answered\n",
            "3. Review your explanations for clarity\n",
            "\n",
            "**To Submit**:\n",
            "- File → Download → Download .ipynb\n",
            "- Submit the notebook file to your course LMS\n",
            "\n",
            "**Note**: Make sure your name is in the filename (e.g., homework_01_yourname.ipynb)"
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
    """Generate all homework notebooks."""
    for hw_num in range(1, 16):
        hw_info = HOMEWORKS[hw_num]
        notebook = create_homework_notebook(hw_num, hw_info)
        
        # Create output directory if it doesn't exist
        output_dir = f"ml2/lecture{hw_num:02d}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Write notebook file
        output_path = f"{output_dir}/homework_{hw_num:02d}.ipynb"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Created {output_path}")
    
    print(f"\\n🎉 Successfully generated all {len(HOMEWORKS)} homework notebooks!")

if __name__ == "__main__":
    main()
