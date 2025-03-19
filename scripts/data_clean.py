#!/usr/bin/env python3

"""
A script to:
1. Recursively find all Markdown files in a 'data' directory.
2. Clean the text (remove markdown symbols, code fences, etc.).
3. Perform basic data exploration (line counts, token counts, etc.).
4. Optionally save cleaned text to disk for downstream tasks.
"""

import os
import re
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

# Make sure you've downloaded the necessary NLTK resources:
# nltk.download('punkt')

def find_markdown_files(root_dir):
    """
    Recursively find all .md files starting from root_dir.
    Returns a list of file paths.
    """
    md_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.md'):
                md_files.append(os.path.join(subdir, file))
    return md_files

def clean_markdown_text(text):
    """
    Basic cleaning of markdown text:
    - Remove code fences/triple backticks
    - Remove other markdown-specific syntax (#, *, etc.)
    - Optionally remove or transform links/images if needed
    """
    # Remove code fences/triple backticks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)

    # Remove inline code backticks
    text = re.sub(r'`[^`]*`', '', text)

    # Remove markdown headings, e.g., # Heading
    text = re.sub(r'#+ ', '', text)

    # Remove images/links of the form ![alt text](url) or [title](url)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)

    # Remove any remaining markdown characters like * or _
    text = text.replace('*', '').replace('_', '')

    # Replace multiple spaces or newlines with a single space
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def basic_exploration(text):
    """
    Performs simple exploration on the cleaned text:
    - Number of lines
    - Number of tokens
    - Unique word count
    - Top 10 most common words (as a sample)
    """
    lines = text.split('\n')
    num_lines = len([l for l in lines if l.strip()])

    # Tokenize using NLTK
    tokens = word_tokenize(text)
    num_tokens = len(tokens)

    # Lowercase tokens for counting
    tokens_lower = [t.lower() for t in tokens if t.isalpha()]
    unique_tokens = set(tokens_lower)
    num_unique = len(unique_tokens)

    # Common words
    counter = Counter(tokens_lower)
    top_common = counter.most_common(10)

    return {
        'num_lines': num_lines,
        'num_tokens': num_tokens,
        'num_unique': num_unique,
        'top_10_common_words': top_common
    }

def main():
    # Change 'data' to your root directory if needed
    root_dir = 'data'
    output_dir = 'cleaned_data'
    os.makedirs(output_dir, exist_ok=True)

    # Locate all .md files
    md_files = find_markdown_files(root_dir)
    print(f"Found {len(md_files)} markdown files.")

    all_clean_text = []
    overall_stats = {
        'total_lines': 0,
        'total_tokens': 0,
        'total_unique_words': set()
    }

    for file_path in md_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        # Clean the text
        cleaned = clean_markdown_text(raw_text)

        # Optionally, you can split the cleaned text back into lines for further analysis
        # lines = cleaned.split('. ')

        # Perform basic exploration
        stats = basic_exploration(cleaned)

        # Aggregate stats
        overall_stats['total_lines'] += stats['num_lines']
        overall_stats['total_tokens'] += stats['num_tokens']
        overall_stats['total_unique_words'].update(
            [w for (w, _) in stats['top_10_common_words']]
        )

        # Print or log stats for each file
        print(f"\nFile: {file_path}")
        print(f"  Lines: {stats['num_lines']}")
        print(f"  Tokens: {stats['num_tokens']}")
        print(f"  Unique Words: {stats['num_unique']}")
        print(f"  Top 10 Common Words: {stats['top_10_common_words']}")

        # Save cleaned text to a new file (optional)
        base_name = os.path.basename(file_path)
        out_path = os.path.join(output_dir, base_name.replace('.md', '_cleaned.txt'))
        with open(out_path, 'w', encoding='utf-8') as out_f:
            out_f.write(cleaned)

        # Also keep track of all text if you want a single aggregated file
        all_clean_text.append(cleaned)

    # Summarize overall stats
    print("\n=== Overall Summary ===")
    print(f"Total lines (across all files): {overall_stats['total_lines']}")
    print(f"Total tokens (across all files): {overall_stats['total_tokens']}")
    print(f"Total unique words (sampled from top freq sets): {len(overall_stats['total_unique_words'])}")

    # Optionally save aggregated text
    aggregated_text = "\n\n".join(all_clean_text)
    with open(os.path.join(output_dir, "all_notes_aggregated.txt"), 'w', encoding='utf-8') as agg_f:
        agg_f.write(aggregated_text)

if __name__ == "__main__":
    main()
