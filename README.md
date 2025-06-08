# Machine Learning Course Materials

This repository contains the course materials for ML1 (Introduction to Machine Learning) and ML2 (Advanced Machine Learning Concepts).

## Project Structure

```
.
├── ml1/                    # Introduction to Machine Learning
│   └── lecture01/         # First lecture
│       └── textbook.md    # Lecture content in markdown
├── ml2/                    # Advanced Machine Learning
│   └── lecture01/         # First lecture
│       └── textbook.md    # Lecture content in markdown
├── output/                 # Generated PDFs (created by script)
├── generate_pdfs.py       # PDF generation script
└── requirements.txt       # Python dependencies
```

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install system dependencies:
- [Pandoc](https://pandoc.org/installing.html)
- [TeX Live](https://www.tug.org/texlive/) or [MacTeX](https://www.tug.org/mactex/) (for PDF generation)

## Usage

To generate PDFs from markdown files:

```bash
python generate_pdfs.py
```

This will:
1. Check for required dependencies
2. Process all markdown files in ml1/ and ml2/
3. Generate PDFs in the output/ directory
4. Maintain course structure in output

## Features

- Parallel processing for faster PDF generation
- Table of contents generation
- Math equation support
- Code syntax highlighting
- Consistent styling across documents
- Orange accent colors for links

## Course Structure

### ML1: Introduction to Machine Learning
- Fundamental concepts
- Basic mathematical foundations
- Simple model implementations

### ML2: Advanced Machine Learning
- Advanced architectures
- Modern training techniques
- State-of-the-art implementations

## Contributing

To add new lectures:
1. Create a new directory: `mlX/lectureXX/`
2. Add markdown content in `textbook.md`
3. Run the PDF generation script 