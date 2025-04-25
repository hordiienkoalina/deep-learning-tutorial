# Project Ideator from Markdown Notes
This project fine-tunes a GPT‑2 model on your local Markdown notes, then generates personalized project ideas using snippets from those notes. It also includes an MLflow deployment script for model logging and serving.

## Overview
1. **Data Cleaning & Aggregation:** Cleans all `.md` files in the `data/` folder and aggregates the text into `cleaned_data/all_notes_aggregated.txt`.
2. **Example Generation:** Uses the aggregated text to generate structured project idea examples and saves them to `examples.jsonl`.
3. **Prepare Training Data & Fine-Tune:** Converts `examples.jsonl` to a plain-text file (`train.txt`) and fine-tunes GPT-2 Medium on this data, saving the model to `gpt2-finetuned`.
4. **Inference:** Generates project ideas by sampling a random snippet from the aggregated notes using the fine-tuned model.
5. **Deployment (MLflow):** Logs the model with MLflow and provides an example script for serving it locally.

## Setup
Create and activate a Python virtual environment:
```
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```

## Run the Pipeline

### 1. Data Cleaning & Aggregation
Upload all your `.md` notes into `data/`.  
Then, clean and aggregate the notes:
```
python3 scripts/clean_notes.py
```

### 2. Example Generation
Generate structured project idea examples from your aggregated notes using either random or non-overlapping sampling:
```
python3 scripts/gen_examples.py --agg_file cleaned_data/all_notes_aggregated.txt --n_examples 100 --output examples_random.jsonl --engine gpt-4o-mini --sampling_strategy random
python3 scripts/gen_examples.py --agg_file cleaned_data/all_notes_aggregated.txt --n_examples 100 --output examples_nonoverlap.jsonl --engine gpt-4o-mini --sampling_strategy non_overlapping
```
- `--agg_file`: Path to the aggregated notes text file.
- `--n_examples`: Number of examples to generate.
- `--output`: Output JSONL file.
- `--engine`: OpenAI model to use for generation.
- `--sampling_strategy`: Choose `random` (default) for random overlapping snippets, or `non_overlapping` for consecutive, non-overlapping segments.

### 3. Fine-Tune GPT-2 Medium
Convert the generated examples to a plain-text file and fine-tune GPT-2 Medium. Run for each dataset:
```
python3 scripts/fine_tune.py --train --jsonl examples_random.jsonl --txt train_random.txt --model_out gpt2-finetuned-random --sampling_strategy random
python3 scripts/fine_tune.py --train --jsonl examples_nonoverlap.jsonl --txt train_nonoverlap.txt --model_out gpt2-finetuned-nonoverlap --sampling_strategy non_overlapping
```

### 4. Inference (Generate Ideas)
Generate project ideas interactively from your fine-tuned model, specifying the sampling strategy for snippet selection:
```
python3 scripts/gen_ideas.py --agg_file cleaned_data/all_notes_aggregated.txt --model_dir gpt2-finetuned-random --n_ideas 5 --sampling_strategy random
python3 scripts/gen_ideas.py --agg_file cleaned_data/all_notes_aggregated.txt --model_dir gpt2-finetuned-nonoverlap --n_ideas 5 --sampling_strategy non_overlapping
```
- `--agg_file`: Path to the aggregated notes text file.
- `--model_dir`: Path to the fine-tuned GPT-2 model directory.
- `--n_ideas`: Number of ideas to generate.
- `--sampling_strategy`: Sampling strategy for snippet selection.

### 5. MLflow Deployment
Get a free port for MLflow:
```
python3 scripts/port.py
```

Start the MLflow Tracking Server:
```
mlflow server --host 127.0.0.1 --port [PORT_FROM_ABOVE]
```

Log the Model with MLflow:
```
python3 scripts/deploy.py
```

---

**Notes:**
- Adjust the number of examples, batch size, and epochs as needed for your hardware and dataset size.
- See each script's docstring or `--help` flag for more details and options.