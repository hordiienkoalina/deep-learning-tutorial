"""
Script: deploy.py
-----------------
MLflow deployment script for the fine-tuned GPT-2 project idea generator.

Features:
- Defines a custom MLflow PyFunc model for project idea generation.
- Loads the fine-tuned GPT-2 model and tokenizer as an MLflow artifact.
- Supports both Apple Silicon (MPS) and CPU for inference.
- Logs the model to MLflow, registers it, and provides a test prediction.

Usage:
    python scripts/deploy.py

Requirements:
    - MLflow Tracking Server running (see README for instructions)
    - Fine-tuned model directory (default: ./gpt2-finetuned-project-ideas)
    - requirements.txt listing all dependencies
"""
import mlflow
import mlflow.pyfunc
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ----------------------------
# 1) Custom MLflow PyFunc Model
# ----------------------------
class ProjectIdeaGenerator(mlflow.pyfunc.PythonModel):
    """
    MLflow PyFunc model for generating project ideas from prompts using a fine-tuned GPT-2 model.
    Loads the model and tokenizer from the provided artifact directory.
    """
    def load_context(self, context):
        # The artifact directory is stored in context.artifacts
        model_dir = context.artifacts["model_dir"]
        # Load tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        # Avoid warnings by setting pad_token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        # Decide on device (MPS for Apple Silicon, or CPU)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)

    def predict(self, context, model_input):
        """
        Expects model_input to be a DataFrame with a "prompt" column.
        Returns a string with the generated project idea.
        """
        prompt = model_input["prompt"].iloc[0]  # just take first row's prompt
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        # Generate text
        outputs = self.model.generate(
            inputs,
            max_length=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            no_repeat_ngram_size=2,
            early_stopping=True,
            num_return_sequences=1
        )
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the start if present
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        return generated_text

# ----------------------------
# 2) MLflow Logging and Testing
# ----------------------------
if __name__ == "__main__":
    """
    Main CLI entry point. Logs the fine-tuned model to MLflow, registers it, and runs a test prediction.
    """
    # Path to the fine-tuned GPT-2 model directory
    model_dir = "./gpt2-finetuned-project-ideas"

    # Specify a custom MLflow Tracking URI (see README for port setup)
    mlflow.set_tracking_uri("http://127.0.0.1:61828")

    # Set (or create) an MLflow experiment
    mlflow.set_experiment("GPT2 Project Idea Generator Experiment")

    # Prepare an input example for model signature
    input_example = pd.DataFrame({
        "prompt": [
            "Based on my notes: Here is a snippet from my design notes about sustainability.\nProject Idea:"
        ]
    })

    # Log artifacts (model directory)
    artifacts = {"model_dir": model_dir}

    with mlflow.start_run() as run:
        # Optionally log any params/metrics/tags
        mlflow.set_tag("Training Info", "GPT-2 project idea generator (fine-tuned).")
        # Log our custom pyfunc model
        model_info = mlflow.pyfunc.log_model(
            artifact_path="project_idea_generator",
            python_model=ProjectIdeaGenerator(),
            artifacts=artifacts,
            pip_requirements="./requirements.txt",  # or the correct path to your file
            input_example=input_example,
            registered_model_name="gpt2-project-idea-generator"
        )
        run_id = run.info.run_id
        print(f"Model logged under run_id: {run_id}")

    # ----------------------------
    # 3) Load the Model for Testing
    # ----------------------------
    logged_model_uri = f"runs:/{run_id}/project_idea_generator"
    loaded_model = mlflow.pyfunc.load_model(logged_model_uri)

    # Make a test prediction
    test_prompt_df = pd.DataFrame({
        "prompt": [
            "Based on my notes: I'm interested in an AI-driven approach for analyzing large sets of urban planning data.\nProject Idea:"
        ]
    })
    output = loaded_model.predict(test_prompt_df)
    print("Generated Project Idea:")
    print(output)
