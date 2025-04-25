"""
Script: fine_tune.py
-------------------
Fine-tunes a GPT-2 model on structured project idea examples.

Features:
- Loads a dataset of structured examples (JSONL) for fine-tuning.
- Fine-tunes GPT-2 (or GPT-2 Medium) on the examples.
- Saves the fine-tuned model and tokenizer.

Usage:
    python scripts/fine_tune.py --train

Arguments:
    --train: Fine-tune GPT-2 on examples.jsonl

Notes:
    run `pip install --upgrade transformers accelerate` if you encounter issues with the accelerate library
"""
import os
import argparse
from pathlib import Path
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import torch

EXAMPLES_PATH = "examples.jsonl"
TRAIN_TXT_PATH = "train.txt"
MODEL_OUT = "gpt2-finetuned"

def prepare_train_txt():
    """
    Converts examples.jsonl to train.txt, one example per line (concatenated fields).
    """
    dataset = load_dataset('json', data_files=EXAMPLES_PATH, split='train')
    def concat_fields(example):
        return {'text': ' '.join(str(v) for v in example.values() if isinstance(v, str))}
    dataset = dataset.map(concat_fields)
    dataset = dataset.remove_columns([col for col in dataset.column_names if col != 'text'])
    with open(TRAIN_TXT_PATH, 'w') as f:
        for ex in dataset['text']:
            f.write(ex + '\n')
    print(f"Prepared {TRAIN_TXT_PATH} from {EXAMPLES_PATH}")

def fine_tune():
    """
    Fine-tune GPT-2 (or GPT-2 Medium) on structured examples from train.txt.
    Saves the fine-tuned model and tokenizer to MODEL_OUT.
    """
    prepare_train_txt()
    # Load tokenizer and model (GPT-2 Medium for more capacity)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    # Prepare dataset and data collator
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=TRAIN_TXT_PATH,
        block_size=128
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_OUT,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        prediction_loss_only=True,
        fp16=torch.cuda.is_available(),
    )
    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)
    print(f"Model fine-tuned and saved to {MODEL_OUT}")

def main():
    """
    Main CLI entry point. Fine-tunes the model based on arguments.
    """
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on examples.jsonl.")
    parser.add_argument('--train', action='store_true', help='Fine-tune GPT-2 on examples.jsonl')
    args = parser.parse_args()

    if args.train:
        fine_tune()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
