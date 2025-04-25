"""
Script: gen_examples.py
----------------------
Generates structured project idea examples from random text snippets using the OpenAI API.

Features:
- Loads a large text file of aggregated notes.
- Samples random snippets from the text.
- Uses the OpenAI API with structured output (JSON schema) to generate project ideas for each snippet.
- Saves the snippet and the structured idea as a JSONL file for fine-tuning language models.

Usage:
    python scripts/gen_examples.py \
        --agg_file cleaned_data/all_notes_aggregated.txt \
        --n_examples 100 \
        --output examples.jsonl \
        --engine gpt-4o-mini

Arguments:
    --agg_file: Path to the aggregated notes text file.
    --n_examples: Number of examples to generate (default: 100).
    --output: Output JSONL file (default: examples.jsonl).
    --engine: OpenAI model to use for generation (default: gpt-4o-mini).
"""
import os
import random
import time
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
import openai

# Load environment variables from .env
load_dotenv()

# Set API key from .env (make sure your .env has OPEN_AI_KEY)
openai.api_key = os.getenv('OPEN_AI_KEY')
if not openai.api_key:
    raise ValueError("Set your OPEN_AI_KEY environment variable in .env first.")

# 1) Define the JSON schema for the structured output
schema = {
    "type": "object",
    "properties": {
        "keywords": {
            "type": "string",
            "description": "Two or three keywords (you may add new ones and/ot pick out very unique random words in the string) representing the intersection of fields."
        },
        "user_problem": {
            "type": "string",
            "description": "A one-sentence statement of the core problem your project solves."
        },
        "description": {
            "type": "string",
            "description": "A 2-3 sentence project idea, referencing the keywords."
        }
    },
    "required": ["keywords", "user_problem", "description"]
}

# 2) System instructions for prompting
system_instructions = """
Take inspiration from each snippet and turn it into a project idea.
It could be a research idea, a software project idea, or a data project idea,
or a creative project idea, or something else—whichever you see fit. The proposed projects should be
doable as personal independent projects or small team portfolio pieces or hobbies.
Be as creative and unique as possible. You can and should add new keywords
if they are not in the snippet to make the projects more interesting.
Projects should lie at the intersection of 2–3 fields. Be as specific as possible 
and genearte niche unique interesteding ideas. Always be confident in your answer. 
You must not mention the tools that should be used to buikd the project. 
"""

def load_aggregated_text(path: Path) -> str:
    """Load the entire contents of a text file as a string."""
    with path.open('r', encoding='utf-8') as f:
        return f.read()

def get_random_snippet(text: str, length: int = 500) -> str:
    """
    Randomly sample a substring (snippet) of the given length from the input text.
    If the text is shorter than the requested length, return the whole text.
    """
    if len(text) <= length:
        return text
    start = random.randint(0, len(text) - length)
    return text[start:start + length]

def generate_example(snippet: str, engine: str = 'gpt-3.5-turbo') -> dict:
    """
    Calls the OpenAI chat endpoint with response_format to force a JSON output
    matching our schema. Returns a dict with 'keywords', 'user_problem', and 'description'.
    """
    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user",   "content": f"Snippet:\n\"{snippet}\""}
    ]
    resp = openai.chat.completions.create(
        model=engine,
        messages=messages,
        temperature=0.7,
        max_tokens=300,
        top_p=0.9,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "ProjectIdea",
                "schema": schema
            }
        }
    )
    # The model’s JSON output lives in content
    return json.loads(resp.choices[0].message.content)

def main():
    """
    Main CLI entry point. Loads data, generates snippet→idea pairs, and saves them as JSONL for fine-tuning.
    """
    parser = argparse.ArgumentParser(
        description="Generate snippet→idea pairs via OpenAI API with Structured Outputs."
    )
    parser.add_argument(
        '--agg_file', type=Path, required=True,
        help='Path to aggregated notes text file'
    )
    parser.add_argument(
        '--n_examples', type=int, default=100,
        help='Number of examples to generate'
    )
    parser.add_argument(
        '--output', type=Path, default=Path('examples.jsonl'),
        help='Output JSONL file'
    )
    parser.add_argument(
        '--engine', type=str, default='gpt-4o-mini',
        help='OpenAI model to use for generation'
    )
    args = parser.parse_args()

    # Load the aggregated notes text
    text = load_aggregated_text(args.agg_file)

    # Open the output file and generate examples
    with args.output.open('w', encoding='utf-8') as out_f:
        for i in range(args.n_examples):
            snippet = get_random_snippet(text)
            try:
                example = generate_example(snippet, engine=args.engine)
                # Merge snippet with the structured response
                record = {
                    "snippet": snippet,
                    **example
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"{i+1}/{args.n_examples} examples generated.")
                time.sleep(1)  # respect rate limits
            except Exception as e:
                print(f"Error generating example {i+1}: {e}")
                break

    print(f"Done! Examples saved to {args.output}")

if __name__ == '__main__':
    main()
