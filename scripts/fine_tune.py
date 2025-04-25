"""
Script: fine_tune.py
-------------------
Fine-tunes a GPT-2 model on structured project idea examples.

Features:
- Loads a dataset of structured examples (JSONL) for fine-tuning.
- Fine-tunes GPT-2 (or GPT-2 Medium) on the examples.
- Saves the fine-tuned model and tokenizer.

Usage:
    python scripts/fine_tune.py --train --jsonl <path_to_jsonl> --txt <path_to_txt> --model_out <output_dir> --sampling_strategy <strategy>

Arguments:
    --train: Fine-tune GPT-2 on the specified dataset
    --search_hyperparams: Run a basic hyperparameter search and log all runs to MLflow
    --jsonl: Path to input examples JSONL file
    --txt: Path to output train.txt file
    --model_out: Output directory for the fine-tuned model
    --sampling_strategy: Sampling strategy used (random or non_overlapping)

Notes:
    run `pip install --upgrade transformers accelerate` if you encounter issues with the accelerate library
"""
import os
import argparse
from pathlib import Path
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import torch
import mlflow

def prepare_train_txt(jsonl_path, txt_path):
    """
    Converts a JSONL file to a TXT file, one example per line (concatenated fields).
    """
    dataset = load_dataset('json', data_files=str(jsonl_path), split='train')
    def concat_fields(example):
        return {'text': ' '.join(str(v) for v in example.values() if isinstance(v, str))}
    dataset = dataset.map(concat_fields)
    dataset = dataset.remove_columns([col for col in dataset.column_names if col != 'text'])
    with open(txt_path, 'w') as f:
        for ex in dataset['text']:
            f.write(ex + '\n')
    print(f"Prepared {txt_path} from {jsonl_path}")
    return dataset

def fine_tune(jsonl_path, txt_path, model_out, sampling_strategy, num_train_epochs=3, per_device_train_batch_size=2, learning_rate=5e-5):
    """
    Fine-tune GPT-2 (or GPT-2 Medium) on structured examples from txt_path.
    Logs parameters and metrics to MLflow.
    """
    dataset = prepare_train_txt(jsonl_path, txt_path)
    num_samples = len(dataset)
    avg_len = sum(len(t) for t in dataset['text']) / num_samples if num_samples > 0 else 0

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=txt_path,
        block_size=128
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    training_args = TrainingArguments(
        output_dir=model_out,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        prediction_loss_only=True,
        fp16=torch.cuda.is_available(),
    )
    with mlflow.start_run():
        mlflow.log_param("sampling_strategy", sampling_strategy)
        mlflow.log_param("num_samples", num_samples)
        mlflow.log_param("avg_sample_length", avg_len)
        mlflow.log_param("model_size", "gpt2-medium")
        mlflow.log_param("train_txt_path", str(txt_path))
        mlflow.log_param("jsonl_path", str(jsonl_path))
        mlflow.log_param("num_train_epochs", num_train_epochs)
        mlflow.log_param("per_device_train_batch_size", per_device_train_batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )
        train_result = trainer.train()
        trainer.save_model(model_out)
        tokenizer.save_pretrained(model_out)
        final_loss = train_result.training_loss if hasattr(train_result, 'training_loss') else None
        mlflow.log_metric("final_training_loss", final_loss)
        print(f"Model fine-tuned and saved to {model_out}")
        print(f"MLflow run logged.")

def hyperparam_search(jsonl_path, txt_path, model_out, sampling_strategy):
    """
    Run a basic grid search over hyperparameters and log each run to MLflow.
    """
    # Define a small grid
    epoch_grid = [2, 3]
    batch_grid = [2, 4]
    lr_grid = [5e-5, 1e-4]
    run_idx = 0
    for num_train_epochs in epoch_grid:
        for per_device_train_batch_size in batch_grid:
            for learning_rate in lr_grid:
                run_idx += 1
                print(f"\n[Hyperparam Search] Run {run_idx}: epochs={num_train_epochs}, batch_size={per_device_train_batch_size}, lr={learning_rate}")
                # Use a unique output dir for each run
                model_out_run = f"{model_out}_search_e{num_train_epochs}_b{per_device_train_batch_size}_lr{str(learning_rate).replace('.', '')}"
                fine_tune(
                    jsonl_path,
                    txt_path,
                    model_out_run,
                    sampling_strategy,
                    num_train_epochs=num_train_epochs,
                    per_device_train_batch_size=per_device_train_batch_size,
                    learning_rate=learning_rate
                )

def main():
    """
    Main CLI entry point. Fine-tunes the model based on arguments.
    """
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on a specified dataset.")
    parser.add_argument('--train', action='store_true', help='Fine-tune GPT-2 on the specified dataset')
    parser.add_argument('--search_hyperparams', action='store_true', help='Run a basic hyperparameter search and log all runs to MLflow')
    parser.add_argument('--jsonl', type=str, required=True, help='Path to input examples JSONL file')
    parser.add_argument('--txt', type=str, required=True, help='Path to output train.txt file')
    parser.add_argument('--model_out', type=str, required=True, help='Output directory for the fine-tuned model')
    parser.add_argument('--sampling_strategy', type=str, required=True, help='Sampling strategy used (random or non_overlapping)')
    args = parser.parse_args()

    if args.search_hyperparams:
        hyperparam_search(args.jsonl, args.txt, args.model_out, args.sampling_strategy)
    elif args.train:
        fine_tune(args.jsonl, args.txt, args.model_out, args.sampling_strategy)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
