# Machine Learning Course Website

This directory contains a comprehensive machine learning course website with multiple courses and topics.

## Directory Structure

### Main Courses
- **ml1/** - Introduction to Machine Learning (11 lectures: 01-07, 10, 12-14)
  - Basic ML concepts and foundations
  - Each lecture contains index.html with course content

- **ml2/** - Advanced Machine Learning (15 lectures: 01-15)
  - Advanced topics and techniques
  - Each lecture includes:
    - index.html (main lecture page)
    - markdown textbooks (ml2_weekXX.md)
    - Jupyter notebooks (homework and exercises)
    - assets/ folder with images and resources

### Topic Libraries
- **mltopics/** - 10 standalone ML topics with textbook content
- **topics/** - Extensive topic library (32+ topics) including:
  - autoencoders, automl, backpropagation, bayesian_methods
  - boosting_bagging, cnn_architectures, cross_validation
  - decision_trees, dimensionality_reduction, GANs
  - gradient_descent, k_nearest_neighbors, learning_rate_schedulers
  - linear_regression, LSTM, neural_networks, regularization
  - recommendation_systems, search_algorithms, text_embeddings
  - time_series, transformers, t-SNE, UMAP, and more
  - Contains template files and generation prompts

### Supporting Directories
- **python/** - Python programming lectures (4 lectures)
- **output/** - Generated PDF outputs organized by course
- **venv/** - Python virtual environment

## Key Scripts

### Content Generation
- `generate_pdfs.py` - Converts markdown textbooks to PDFs
- `generate_ml2_homeworks.py` - Generates ML2 homework assignments (117KB)
- `generate_ml2_notebooks.py` - Creates ML2 Jupyter notebooks
- `update_ml2_pages.py` - Updates ML2 lecture pages

### Content Management
- `add_homework_buttons.py` - Adds homework UI buttons to pages
- `course_generator_prompt.md` - Template for course content generation

## Web Files
- `index.html` - Main website homepage
- `resource-library.html` - Resource library page (326KB)
- `slideformat.txt` - Slide formatting guidelines

## Configuration
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore rules
- `README.md` - Project documentation

## Workflow
1. Write course content in markdown files
2. Use generation scripts to create homework/exercises
3. Run `generate_pdfs.py` to create PDF versions
4. Update pages with `update_ml2_pages.py`
5. Add interactive elements with `add_homework_buttons.py`
