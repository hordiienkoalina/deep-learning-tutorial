"""
Script: gen_ideas.py
-------------------
Generates project ideas from random text snippets using a fine-tuned GPT-2 model.

Features:
- Loads a large text file of aggregated notes.
- Samples random snippets from the text.
- Uses a fine-tuned GPT-2 model to generate project ideas based on each snippet.
- Prints the snippet and the generated idea to the CLI.
- Applies decoding strategies and post-processing to improve output quality.

Usage:
    python scripts/gen_ideas.py \
        --agg_file cleaned_data/all_notes_aggregated.txt \
        --model_dir gpt2-finetuned \
        --n_ideas 5

Arguments:
    --agg_file: Path to the aggregated notes text file.
    --model_dir: Path to the fine-tuned GPT-2 model directory.
    --n_ideas: Number of ideas to generate (default: 5).
    --snippet_len: Length of each random snippet (default: 500).
"""
import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re


def load_text(path: Path) -> str:
    """Load the entire contents of a text file as a string."""
    return path.read_text(encoding='utf-8')


def sample_snippet(text: str, length: int = 500) -> str:
    """
    Randomly sample a substring (snippet) of the given length from the input text.
    If the text is shorter than the requested length, return the whole text.
    """
    if len(text) <= length:
        return text
    start = torch.randint(0, len(text) - length, (1,)).item()
    return text[start:start + length]


def get_non_overlapping_snippets(text: str, length: int = 500, n_examples: int = 5):
    """
    Generate non-overlapping snippets of the given length from the input text.
    Returns up to n_examples snippets.
    """
    snippets = []
    for start in range(0, len(text), length):
        if len(snippets) >= n_examples:
            break
        end = start + length
        snippet = text[start:end]
        if snippet:
            snippets.append(snippet)
    return snippets


def parse_output(output: str) -> dict:
    """
    Attempt to extract 'keywords', 'user_problem', and 'description' fields from the model output using regex.
    Prints the raw model output for debugging.
    Returns a dict with the extracted fields (may be empty if not found).
    """
    result = {'keywords': '', 'user_problem': '', 'description': ''}
    print("DEBUG: Model output:", repr(output))
    # Regex is case-insensitive and robust to whitespace
    keywords = re.search(r"(?i)keywords\s*:\s*(.*?)(problem\s*:|description\s*:|$)", output, re.DOTALL)
    problem = re.search(r"(?i)problem\s*:\s*(.*?)(description\s*:|$)", output, re.DOTALL)
    description = re.search(r"(?i)description\s*:\s*(.*)", output, re.DOTALL)
    if keywords:
        result['keywords'] = keywords.group(1).strip()
    if problem:
        result['user_problem'] = problem.group(1).strip()
    if description:
        result['description'] = description.group(1).strip()
    return result


def generate_idea_with_retry(prompt, model, tokenizer, max_retries=3, min_fields=2):
    """
    Generate a project idea from a prompt using the model, retrying up to max_retries times if the output is incomplete.
    Returns a dict with 'keywords', 'user_problem', and 'description' fields (may be 'N/A' if not found).
    """
    for _ in range(max_retries):
        # Tokenize the prompt and generate output
        inputs = tokenizer(prompt, return_tensors='pt', padding=True)
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,  # Allow up to 120 new tokens for the idea
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.8,
            top_k=40,
            temperature=0.5,  # Lower temperature for more coherent endings
            repetition_penalty=1.2,
            num_return_sequences=1,
            early_stopping=True
        )
        # Decode and post-process the output
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_part = generated[len(prompt):].strip()
        # Trim to last full sentence for cleaner output
        def trim_to_last_sentence(text):
            sentences = re.split(r'(?<=[.!?]) +', text)
            if len(sentences) > 1:
                return ' '.join(sentences[:-1]) if not sentences[-1].endswith(('.', '!', '?')) else text
            return text
        generated_part = trim_to_last_sentence(generated_part)
        parsed = parse_output(generated_part)
        # If enough fields are found, return the parsed result
        if sum(bool(parsed[k]) for k in parsed) >= min_fields:
            return parsed
    # If still incomplete after retries, fill missing fields with 'N/A'
    return {k: v if v else 'N/A' for k, v in parsed.items()}


def main():
    """
    Main CLI entry point. Loads data, model, and generates project ideas from random snippets.
    """
    load_dotenv()
    parser = argparse.ArgumentParser("Generate ideas via fine-tuned GPT-2 CLI")
    parser.add_argument('--agg_file', type=Path, required=True,
                        help='Aggregated notes text file')
    parser.add_argument('--model_dir', type=Path, required=True,
                        help='Path to fine-tuned GPT-2 model directory')
    parser.add_argument('--n_ideas', type=int, default=5,
                        help='Number of ideas to generate')
    parser.add_argument('--snippet_len', type=int, default=500,
                        help='Length of each random snippet')
    parser.add_argument('--sampling_strategy', type=str, default='random', choices=['random', 'non_overlapping'],
                        help='Sampling strategy for snippet selection: random (default, overlapping) or non_overlapping (consecutive, non-overlapping segments)')
    args = parser.parse_args()

    # Load the aggregated notes text
    text = load_text(args.agg_file)
    # Load the fine-tuned GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(str(args.model_dir))
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(str(args.model_dir))
    model.eval()

    # Select snippets according to the chosen strategy
    if args.sampling_strategy == 'random':
        snippets = [sample_snippet(text, args.snippet_len) for _ in range(args.n_ideas)]
    else:
        snippets = get_non_overlapping_snippets(text, args.snippet_len, args.n_ideas)
        # If not enough non-overlapping snippets, pad with random
        if len(snippets) < args.n_ideas:
            extra_needed = args.n_ideas - len(snippets)
            snippets += [sample_snippet(text, args.snippet_len) for _ in range(extra_needed)]

    for i, snippet in enumerate(snippets, 1):
        # Construct a minimal, zero-shot prompt to encourage the model to use the snippet
        prompt = (
            f"Snippet: {snippet}\n"
            "Based on the above snippet, generate a project idea in the following format:\n"
            "Keywords:"
        )
        # Ensure prompt + generated tokens <= 1024 (GPT-2 context window)
        gen_tokens = 100
        max_length = 1024
        max_prompt_tokens = max_length - gen_tokens
        prompt_tokens = tokenizer(prompt, return_tensors='pt', padding=True, truncation=False)['input_ids'][0]
        if len(prompt_tokens) > max_prompt_tokens:
            # Truncate the prompt from the start if needed
            prompt_tokens = prompt_tokens[-max_prompt_tokens:]
            prompt = tokenizer.decode(prompt_tokens)
        # Generate the idea from the model
        inputs = tokenizer(prompt, return_tensors='pt', padding=True)
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,  # Allow up to 120 new tokens for the idea
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.8,
            top_k=40,
            temperature=0.5,  # Lower temperature for more coherent endings
            repetition_penalty=1.2,
            num_return_sequences=1,
            early_stopping=True
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_part = generated[len(prompt):].strip()
        # Trim to last full sentence for cleaner output
        def trim_to_last_sentence(text):
            sentences = re.split(r'(?<=[.!?]) +', text)
            if len(sentences) > 1:
                return ' '.join(sentences[:-1]) if not sentences[-1].endswith(('.', '!', '?')) else text
            return text
        generated_part = trim_to_last_sentence(generated_part)
        # Print the snippet and the generated idea
        print(f"\n=== Idea {i}/{args.n_ideas} ===")
        print(f"Snippet: {snippet} \n")
        print(f"Idea: {generated_part}")

if __name__ == '__main__':
    main()
