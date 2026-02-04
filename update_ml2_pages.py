#!/usr/bin/env python3
"""
Update all ML2 weekly pages to professional format.
Removes emoticons and colored buttons, adds professional text links at top.
"""

import os
import re
from pathlib import Path

# Base directory for ML2 lectures
ML2_BASE = Path("/Users/john/Dropbox/_______Cursor/ml_website_2/ml2")

# CSS to add for top-resources
TOP_RESOURCES_CSS = """        /* Top Resources Links */
        .top-resources {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            padding: 20px 0;
            margin-bottom: 20px;
            border-bottom: 1px solid #eee;
        }

        .resource-btn {
            padding: 8px 16px;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 600;
            text-decoration: none;
            transition: all 0.2s;
            background-color: white;
            color: #354CA1;
            border: 1px solid #354CA1;
            text-align: center;
        }

        .resource-btn:hover {
            background-color: #f0f4ff;
            text-decoration: none;
        }

"""

def extract_resource_links(content, week_num):
    """Extract resource links from the old button format."""
    resources = {}
    
    # Find video link
    video_match = re.search(r'href="([^"]*)"[^>]*class="[^"]*video[^"]*"', content, re.IGNORECASE)
    if video_match:
        resources['videos'] = video_match.group(1)
    else:
        resources['videos'] = '#'
    
    # Find textbook link
    textbook_match = re.search(r'href="([^"]*\.pdf)"[^>]*class="[^"]*textbook[^"]*"', content, re.IGNORECASE)
    if textbook_match:
        resources['textbook'] = textbook_match.group(1)
    else:
        resources['textbook'] = f'../../output/ml2/lecture{week_num:02d}/ml2_week{week_num:02d}.pdf'
    
    # Find colab links
    colab_matches = re.findall(r'href="(https://colab\.research\.google\.com/[^"]*)"', content)
    resources['colabs'] = colab_matches if colab_matches else []
    
    return resources

def create_top_resources_html(resources, week_num):
    """Create the HTML for the top resources section."""
    html_parts = []
    html_parts.append('            <div class="top-resources">')
    html_parts.append(f'                <a href="{resources["videos"]}" class="resource-btn">Videos</a>')
    html_parts.append(f'                <a href="{resources["textbook"]}" class="resource-btn">Textbook</a>')
    
    # Add colab links
    for i, colab_url in enumerate(resources['colabs']):
        if 'exercises' in colab_url.lower() or 'week_' in colab_url.lower():
            html_parts.append(f'                <a href="{colab_url}" class="resource-btn" target="_blank">Colab</a>')
        elif 'homework' in colab_url.lower():
            # Save homework for last
            continue
        else:
            # Additional colab notebooks
            html_parts.append(f'                <a href="{colab_url}" class="resource-btn" target="_blank">Colab {i+1}</a>')
    
    # Add homework link at the end
    for colab_url in resources['colabs']:
        if 'homework' in colab_url.lower():
            html_parts.append(f'                <a href="{colab_url}" class="resource-btn" target="_blank">Homework</a>')
            break
    
    html_parts.append('            </div>')
    return '\n'.join(html_parts)

def update_week_page(week_num):
    """Update a single week's page to the new format."""
    print(f"Updating Week {week_num}...")
    
    file_path = ML2_BASE / f"lecture{week_num:02d}" / "index.html"
    
    if not file_path.exists():
        print(f"  ⚠️  File not found: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract resource links
    resources = extract_resource_links(content, week_num)
    
    # Step 1: Add top-resources CSS if not already present
    if 'top-resources' not in content:
        # Find where to insert CSS (after a:hover or before .resources)
        css_insert_pattern = r'(        a:hover \{[^}]+\})\n\n(        /\* Resources Section)'
        if re.search(css_insert_pattern, content):
            content = re.sub(
                css_insert_pattern,
                r'\1\n\n' + TOP_RESOURCES_CSS + r'\2',
                content,
                count=1
            )
    
    # Step 2: Remove old button CSS (.action-buttons, .btn-action, .btn-textbook, .btn-colab)
    # Find and remove the action-buttons CSS block
    content = re.sub(
        r'\n\s*\.action-buttons \{[^}]+\}\n\n\s*\.btn-action \{[^}]+\}\n\n\s*\.btn-textbook \{[^}]+\}\n\s*\.btn-textbook:hover \{[^}]+\}\n\n\s*\.btn-colab \{[^}]+\}\n\s*\.btn-colab:hover \{[^}]+\}\n',
        '\n',
        content,
        flags=re.DOTALL
    )
    
    # Step 3: Add top-resources HTML after h1
    if '<div class="top-resources">' not in content:
        top_resources_html = create_top_resources_html(resources, week_num)
        
        # Insert after <h1>Week X:...</h1>
        h1_pattern = r'(<h1>Week \d+:[^<]+</h1>)\n'
        content = re.sub(
            h1_pattern,
            r'\1\n\n' + top_resources_html + '\n',
            content,
            count=1
        )
    
    # Step 4: Remove old action-buttons div with emoticons at the bottom
    # This includes the entire <div class="action-buttons">...</div> block
    content = re.sub(
        r'\n\s*<div class="action-buttons">.*?</div>\n',
        '\n',
        content,
        flags=re.DOTALL
    )
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  ✓ Week {week_num} updated successfully")
    return True

def main():
    """Update all ML2 weeks except week 5 (already done)."""
    print("Updating ML2 weekly pages to professional format...")
    print("=" * 60)
    
    updated = []
    skipped = []
    
    for week_num in range(1, 16):
        if week_num == 5:
            print(f"Week {week_num}: ✓ Already updated (skipping)")
            continue
        
        try:
            if update_week_page(week_num):
                updated.append(week_num)
            else:
                skipped.append(week_num)
        except Exception as e:
            print(f"  ❌ Error updating Week {week_num}: {e}")
            skipped.append(week_num)
    
    print("=" * 60)
    print(f"✅ Successfully updated {len(updated)} weeks")
    if skipped:
        print(f"⚠️  Skipped {len(skipped)} weeks: {skipped}")
    print("=" * 60)

if __name__ == "__main__":
    main()
