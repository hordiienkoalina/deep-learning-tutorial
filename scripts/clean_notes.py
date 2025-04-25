"""
Script: clean_notes.py
---------------------
Cleans and aggregates markdown notes for downstream ML tasks.

Features:
- Recursively finds all .md files in the data/ directory.
- Cleans markdown text (removes code, markdown, stopwords, etc.).
- Saves each cleaned note to cleaned_data/.
- Aggregates all cleaned notes into cleaned_data/all_notes_aggregated.txt.

Usage:
    python scripts/clean_notes.py
"""
import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ----------------------------
# Data Cleaning and Aggregation
# ----------------------------

# Define custom words to remove
CUSTOM_STOPWORDS = set([
    "png", "screenshot", "doi", "https", "screenshot2021",
    "project", "capstone", "work", "assignment", "class", "round",
    "hcs", "feedback", "hc", "untitled", "pcw", "summary",
    "guide", "readings", "study", "learning", "outcomes", "notes"
])
# Combine with NLTK's English stopwords
try:
    ALL_STOPWORDS = set(stopwords.words('english')).union(CUSTOM_STOPWORDS)
except LookupError:
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    ALL_STOPWORDS = set(stopwords.words('english')).union(CUSTOM_STOPWORDS)

def find_markdown_files(root_dir):
    """Recursively find all .md files starting from root_dir."""
    md_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.md'):
                md_files.append(os.path.join(subdir, file))
    return md_files

def clean_markdown_text(text):
    """
    Clean markdown text:
      - Remove code fences, inline code, markdown headings, images/links, and markdown symbols.
      - Replace multiple spaces/newlines with a single space.
      - Tokenize and remove stopwords.
    """
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`[^`]*`', '', text)
    text = re.sub(r'#+ ', '', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = text.replace('*', '').replace('_', '')
    text = re.sub(r'\s+', ' ', text)
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in ALL_STOPWORDS]
    cleaned_text = " ".join(filtered_tokens)
    return cleaned_text.strip()

def aggregate_clean_notes(root_dir='data', output_dir='cleaned_data'):
    """Finds, cleans, and aggregates all markdown notes."""
    os.makedirs(output_dir, exist_ok=True)
    md_files = find_markdown_files(root_dir)
    print(f"Found {len(md_files)} markdown files.")
    all_clean_text = []

    for file_path in md_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        cleaned = clean_markdown_text(raw_text)
        base_name = os.path.basename(file_path)
        out_path = os.path.join(output_dir, base_name.replace('.md', '_cleaned.txt'))
        with open(out_path, 'w', encoding='utf-8') as out_f:
            out_f.write(cleaned)
        all_clean_text.append(cleaned)

    aggregated_text = "\n\n".join(all_clean_text)
    aggregated_file = os.path.join(output_dir, "all_notes_aggregated.txt")
    with open(aggregated_file, 'w', encoding='utf-8') as agg_f:
        agg_f.write(aggregated_text)
    print(f"Aggregated text saved to {aggregated_file}")
    return aggregated_file

if __name__ == '__main__':
    aggregate_clean_notes('data', 'cleaned_data')