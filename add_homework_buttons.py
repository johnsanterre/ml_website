#!/usr/bin/env python3
"""
Add homework buttons to all ML2 lectures that don't have them yet.
"""

import re

# Read the file
with open('index.html', 'r') as f:
    lines = f.readlines()

# Track lectures that need homework buttons
lectures_to_add = []

# Find all ML2 lecture sections
in_ml2_lecture = False
current_lecture = None
has_homework = False

for i, line in enumerate(lines):
    # Detect start of lectures 1-4, 8-12 (those still needing homework buttons)
    if 'ml2/lecture' in line and 'index.html' in line:
        match = re.search(r'ml2/lecture(\d+)/index\.html', line)
        if match:
            current_lecture = int(match.group(1))
            in_ml2_lecture = True
            has_homework = False
    
    # Check if this lecture already has homework button
    if in_ml2_lecture and 'homework-button' in line:
        has_homework = True
    
    # At end of resources div, check if we need to add homework
    if in_ml2_lecture and '</div>' in line and 'resources' in lines[i-5:i]:
        if current_lecture and not has_homework and 1 <= current_lecture <= 15:
            lectures_to_add.append((i, current_lecture))
        in_ml2_lecture = False
        current_lecture = None

# Add homework buttons (in reverse to preserve line numbers)
for line_num, lecture_num in reversed(lectures_to_add):
    lecture_str = f'{lecture_num:02d}'
    homework_button = f'                        <a href="https://colab.research.google.com/github/johnsanterre/ml_website/blob/main/ml2/lecture{lecture_str}/homework_{lecture_str}.ipynb"\\n                            class="homework-button" target="_blank">Homework</a>\\n'
    lines.insert(line_num, homework_button)

# Write back
with open('index.html', 'w') as f:
    f.writelines(lines)

print(f'✓ Added homework buttons to {len(lectures_to_add)} lectures: {[lec for _, lec in lectures_to_add]}')
