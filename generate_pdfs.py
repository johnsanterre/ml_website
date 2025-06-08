import os
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import shutil

def check_dependencies():
    """Check if required external dependencies are installed."""
    dependencies = ['pandoc', 'pdflatex']
    missing = []
    
    for dep in dependencies:
        try:
            subprocess.run([dep, '--version'], capture_output=True)
        except FileNotFoundError:
            missing.append(dep)
    
    if missing:
        print(f"Error: Missing required dependencies: {', '.join(missing)}")
        print("\nPlease install the missing dependencies:")
        print("- pandoc: https://pandoc.org/installing.html")
        print("- pdflatex: Install TeX Live or MacTeX")
        sys.exit(1)

def convert_markdown_to_pdf(markdown_file, output_pdf):
    """Convert a markdown file to PDF using pandoc."""
    try:
        # Ensure output directory exists
        output_dir = Path(output_pdf).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Using pandoc with PDF engine
        cmd = [
            'pandoc',
            str(markdown_file),
            '-o', str(output_pdf),
            '--pdf-engine=pdflatex',
            '--highlight-style=tango',
            '-V', 'geometry:margin=1in',
            '--mathjax',
            '--toc',  # Add table of contents
            '-V', 'colorlinks=true',
            '-V', 'linkcolor=orange',
            '-V', 'toccolor=orange'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Successfully converted {markdown_file}")
            return True
        else:
            print(f"✗ Error converting {markdown_file}")
            print(f"Error output: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Error processing {markdown_file}: {str(e)}")
        return False

def process_course_files(course_dir):
    """Process all markdown files in a course directory."""
    course_path = Path(course_dir)
    if not course_path.exists():
        print(f"Warning: Course directory {course_dir} does not exist. Skipping.")
        return []
    
    tasks = []
    for lecture_dir in sorted(course_path.glob("lecture*")):
        for md_file in lecture_dir.glob("*.md"):
            # Create corresponding PDF path maintaining directory structure
            rel_path = md_file.relative_to(course_path)
            output_pdf = Path("output") / course_path.name / rel_path.with_suffix('.pdf')
            tasks.append((md_file, output_pdf))
    
    return tasks

def main():
    """Main function to process all markdown files."""
    print("Checking dependencies...")
    check_dependencies()
    
    # Create base output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Process both ML1 and ML2 courses
    courses = ["ml1", "ml2"]
    all_tasks = []
    
    for course in courses:
        tasks = process_course_files(course)
        all_tasks.extend(tasks)
    
    if not all_tasks:
        print("No markdown files found to process.")
        return
    
    print(f"\nFound {len(all_tasks)} files to process...")
    
    # Process files in parallel
    with ThreadPoolExecutor() as executor:
        futures = []
        for md_file, pdf_file in all_tasks:
            future = executor.submit(convert_markdown_to_pdf, md_file, pdf_file)
            futures.append((future, md_file))
        
        # Wait for all conversions to complete
        for future, md_file in futures:
            future.result()
    
    print("\nPDF generation complete!")

if __name__ == "__main__":
    main() 