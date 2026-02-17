# ML2: Advanced Machine Learning (Deep Learning)

This directory contains the advanced machine learning course with a focus on deep learning concepts.

## Structure

Contains 15 lectures (lecture01 through lecture15) covering deep learning and advanced ML topics.

## Content Format

Each lecture directory typically contains:
- `index.html` - Main lecture webpage
- `ml2_weekXX.md` - Markdown textbook content with theory and equations
- `homework_XX.ipynb` - Jupyter notebook with homework assignments
- `week_XX_exercises.ipynb` - Practice exercises in Jupyter notebook format
- `assets/` - Images, diagrams, and other supporting resources

## Course Focus

ML2 covers advanced topics including:
- Deep Learning fundamentals and neural network architectures
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs) and LSTMs
- Transformers and attention mechanisms
- Modern training techniques and optimization
- Advanced architectures and state-of-the-art implementations
- Mathematical foundations with detailed equations

## Content Features

- Comprehensive markdown textbooks with LaTeX math equations
- Interactive Jupyter notebooks for hands-on practice
- Homework assignments with coding exercises
- Visual assets and diagrams to support learning
- Theory combined with practical implementations

## Generation & Management

Content is managed by several scripts:
- `/generate_ml2_homeworks.py` - Generates homework notebooks
- `/generate_ml2_notebooks.py` - Creates exercise notebooks
- `/update_ml2_pages.py` - Updates lecture HTML pages
- `/add_homework_buttons.py` - Adds interactive UI elements
- PDFs generated in `/output/ml2/`

## Prerequisites

Students should complete ML1 (Introduction to Machine Learning) before starting ML2.
