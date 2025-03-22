"""
This script does the following:
1. Aggregates and cleans all markdown notes (.md) stored in /data.
2. Prepares training dataset of sample responses for project idea generation.
3. Fine-tunes a pre-trained language model (GPT-2) on that training data.
4. Provides an inference interface (CLI) that generates personalized project ideas using a random snippet from the input.
"""

import os
import re
import pickle
import argparse
import random

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    TextDataset,
    DataCollatorForLanguageModeling,
)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# ----------------------------
# Device Setup
# ----------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ----------------------------
# Data Cleaning and Exploration
# ----------------------------

# Define custom words to remove
CUSTOM_STOPWORDS = set([
    "png", "screenshot", "doi", "https", "screenshot2021",
    "project", "capstone", "work", "assignment", "class", "round",
    "hcs", "feedback", "hc", "untitled", "pcw", "summary",
    "guide", "readings", "study", "learning", "outcomes", "notes"
])
# Combine with NLTK's English stopwords
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

# ----------------------------
# Training Data Preparation
# ----------------------------

def prepare_training_data(aggregated_file, training_file='project_ideas_training.txt'):
    """
    Creates a simple training file with sample responses.
    """
    # Load aggregated notes text
    with open(aggregated_file, 'r', encoding='utf-8') as f:
        aggregated_text = f.read().strip()

    # Local helper: returns a random snippet from the text.
    def get_random_snippet(text, snippet_length=500):
        if len(text) <= snippet_length:
            return text
        start_idx = random.randint(0, len(text) - snippet_length)
        return text[start_idx:start_idx + snippet_length]

    # Define synthetic responses
    responses = [
    "A researcher struggles to quickly sort through voluminous study notes. Develop a chatbot using NLP libraries (spaCy, transformers) to classify, summarize, and tag notes for efficient retrieval.",
    "Students often face difficulties finding precise study materials. Create an intelligent search engine that leverages Elasticsearch and transformer-based summarizers (BERT) to fetch and condense relevant content.",
    "Design teams need inspiration from past projects. Build a recommendation system using topic modeling (LDA) and collaborative filtering to suggest project ideas based on archived design data.",
    "Urban planners lack efficient ways to gather community input. Develop a mobile app that utilizes interactive sketching (via Paper.js) and geolocated surveys to collect participatory design feedback.",
    "City officials require tools to assess how design choices affect community well-being. Create an urban planning tool that integrates GIS mapping (with QGIS/ArcGIS) and social–environmental data visualization.",
    "Students have trouble visualizing the evolution of design theories. Build an interactive timeline web app using D3.js to illustrate key shifts in architectural debates and design discourse.",
    "A design researcher wants to forecast emerging trends. Develop a predictive model employing time-series analysis (LSTM networks) on historical design data to highlight future patterns.",
    "Users are unclear on how iterative design improves outcomes. Create an agent-based simulation (using NetLogo or Mesa) to demonstrate the iterative design thinking process and its impact on technology–society coevolution.",
    "Design teams struggle with planning and retrospectives. Build a web-based agile management tool using Django and React that integrates sprint planning, Kanban boards, and feedback loops.",
    "Developers face performance issues under heavy database loads. Create a scalable application that optimizes SQL transactions through performance tuning and load testing (with Apache JMeter).",
    "Financial analysts face challenges in real-time market analysis. Develop a dashboard that integrates live financial APIs and visualization tools (like Plotly) to monitor stock trends and trading volumes.",
    "System administrators need real-time server diagnostics for troubleshooting. Build an HTTP logging and debugging platform using Flask and Loguru to aggregate and analyze server logs.",
    "Policy analysts struggle to evaluate the impact of urban interventions. Create an analytical tool using synthetic control methods (via Python’s statsmodels or R packages) to assess policy outcomes.",
    "Community designers need to align stakeholder input with project goals. Develop a collaborative platform using React and Firebase that supports real-time whiteboarding and video co-creation sessions.",
    "Architects seek guidance on implementing sustainable design practices. Build a recommendation system that fuses environmental sensor data with cultural datasets to propose eco-friendly architectural strategies.",
    "Urban planners require practical advice on incorporating design justice principles. Develop a GPT-powered chatbot that offers actionable recommendations on equity and inclusivity in urban design.",
    "Designers experience bottlenecks in rapid prototyping workflows. Create an adaptive prototyping tool that leverages Figma’s API and iterative testing to streamline design feedback and iterations.",
    "Project managers lack insight into design research progress. Build an interactive dashboard using Plotly in Python that integrates agile metrics and research outputs to track sprint progress and engagement.",
    "Stakeholders need to understand the impact of technology on urban development. Develop a simulation platform using system dynamics modeling (with Vensim or AnyLogic) to explore coevolutionary trends.",
    "Researchers need a faster way to access and annotate design literature. Create a research assistant tool that employs web scraping (using BeautifulSoup) and academic APIs (like CrossRef) to compile and annotate sources.",
    "Field architects struggle to document on-site observations effectively. Build a mobile app that captures and organizes field notes and photos, incorporating GPS tagging and voice-to-text conversion.",
    "Design critics find it challenging to experience and critique spaces virtually. Develop a VR tool using Unity to simulate urban spaces for immersive spatial analysis and design critique.",
    "Digital platforms lack in-depth insights into user engagement with design content. Create a tool that integrates the Google Analytics API with custom dashboards to analyze user behavior on design sites.",
    "Designers need to convert hand-drawn sketches into editable digital wireframes. Build a computer vision solution using OpenCV to transform scanned sketches into digital mockups.",
    "Creative teams want to leverage archived design data for fresh inspiration. Develop a recommendation engine that mines historical project data to suggest new, data-driven creative concepts.",
    "Interdisciplinary teams struggle to collaborate across design and engineering domains. Create a platform that integrates video conferencing, shared whiteboards, and co-editing tools to streamline collaboration.",
    "Architects often lack a tangible sense of proposed changes during presentations. Build an augmented reality app that overlays digital design proposals onto live urban environments for visualization.",
    "Historians and designers need to digitize and search legacy design documents. Develop a digital archive system that uses OCR (with Tesseract) to convert historical texts into searchable records.",
    "Researchers want to monitor real-time trends in design and architecture. Create a web platform that scrapes social media (using the Twitter API) and applies sentiment analysis to track emerging discussions.",
    "Sustainability experts need to assess the environmental impact of building materials. Build a data analytics tool that processes sensor and lifecycle data to compare material sustainability.",
    "Art and design students often find theoretical concepts too abstract. Develop an interactive learning tool that simplifies complex design theories through multimedia tutorials and quizzes.",
    "Project teams struggle with managing time and resources in design projects. Create a project management solution that incorporates time tracking, task automation, and agile workflow features.",
    "Urban sustainability initiatives are hampered by the absence of real-time environmental data. Build an IoT system that monitors air quality and environmental metrics to inform sustainable planning.",
    "City planners need to analyze pedestrian flow data to improve urban safety. Develop a GIS-based platform that processes and visualizes movement patterns to optimize urban mobility.",
    "Educators seek engaging digital tools to teach design methodology and critical theory. Create an interactive, gamified learning platform that guides users through design thinking case studies.",
    "Ethnographic researchers require efficient tools for organizing and analyzing field data. Build an application that integrates transcription services and thematic coding to process interview recordings.",
    "HCI designers need simulated environments to test and iterate user interfaces. Develop a VR simulation using Unity that models public spaces for evaluating human-computer interactions.",
    "Community curators face challenges documenting local public art installations. Create a mobile app that captures images, records audio narratives, and collects community feedback on artworks.",
    "Energy analysts need to optimize building consumption models. Build a machine learning system that predicts and adjusts energy usage by analyzing sensor data and regression models.",
    "Design critics require a centralized platform to aggregate and analyze peer feedback. Develop a web tool that integrates cloud storage, real-time editing, and annotation for design reviews.",
    "Urban climatologists need to visualize local temperature variations and urban heat islands. Create a dashboard that merges sensor data with community reports using advanced data visualization.",
    "Digital storytellers in architecture need tools to generate compelling project narratives. Develop an automated narrative generator that uses NLP and multimedia templates to craft stories from design data.",
    "Rapid prototyping in software design is hindered by manual coding tasks. Build a system that automates prototype generation using code generation tools and modular design templates.",
    "Design proposal evaluation is often subjective. Create a platform that collects crowdsourced feedback and employs statistical analysis to identify consensus areas in proposals.",
    "Teams struggle to collaborate on dynamic design documents in real time. Develop a web-based collaboration tool that integrates cloud editing and version control for synchronous work.",
    "Urban accessibility assessments are time-consuming. Build a computer vision system using TensorFlow to analyze street-level imagery for accessibility features and compliance.",
    "Social equity in urban planning remains underexplored. Create a simulation using agent-based modeling (with Mesa or NetLogo) to evaluate how different policies affect community equity.",
    "Gamers lack tools to analyze in-game behavior. Develop a game analytics platform that leverages machine learning to optimize gameplay mechanics and enhance player engagement.",
    "Writers struggle to organize creative drafts. Build an AI-assisted writing tool that uses NLP to suggest plot improvements, structure narratives, and manage version control.",
    "Music producers have difficulty managing large audio sample libraries. Create an application that employs audio fingerprinting to tag, organize, and retrieve music samples."
    ]

    # Build training examples with a random snippet for each one.
    examples = []
    for response in responses:
        random_snippet = get_random_snippet(aggregated_text, snippet_length=500)
        prompt = f"Based on my notes: {random_snippet}\nProject Idea:"
        examples.append({
            "prompt": prompt,
            "response": response
        })

    # Write the training examples to the file.
    with open(training_file, 'w', encoding='utf-8') as out_f:
        for ex in examples:
            out_f.write(ex["prompt"] + " " + ex["response"] + "\n\n")
    print(f"Training data saved to {training_file}")
    return training_file

# ----------------------------
# Fine-Tuning a Pre-trained Language Model
# ----------------------------

def fine_tune_model(training_file, model_output_dir="gpt2-finetuned-project-ideas", num_train_epochs=3):
    """
    Fine-tunes GPT-2 using the provided training file.
    """
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # Set pad token to avoid warnings during generation
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Create a dataset and data collator
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=training_file,
        block_size=36  # Adjust based on your training examples
    )
    print("Number of training samples:", len(train_dataset))
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning complete.")
    # Save the fine-tuned model and tokenizer
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    print(f"Model saved to {model_output_dir}")
    return model, tokenizer

# ----------------------------
# Inference: Generating Project Ideas
# ----------------------------

def generate_project_idea(prompt, model, tokenizer, max_length=150, temperature=0.7, top_p=0.9):
    """
    Given a prompt, generate a project idea using the fine-tuned model.
    Ensures that the input tensor is moved to the same device as the model.
    """
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        no_repeat_ngram_size=2,
        early_stopping=True,
        num_return_sequences=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def load_aggregated_text(file_path):
    """Loads the aggregated notes text from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def get_random_snippet(aggregated_text, snippet_length=500):
    """Selects a random snippet from the aggregated text."""
    if len(aggregated_text) <= snippet_length:
        return aggregated_text
    start_idx = random.randint(0, len(aggregated_text) - snippet_length)
    return aggregated_text[start_idx:start_idx + snippet_length]

def cli_interface(model, tokenizer, aggregated_file):
    """
    A command-line interface that prompts the user for the number of project ideas to generate,
    then for each idea, selects a new random snippet from the aggregated notes and prints
    the generated project idea labeled as "Project Idea #X".
    """
    aggregated_text = load_aggregated_text(aggregated_file)
    
    print("\n--- Project Idea Generator ---")
    num_input = input("Enter the number of project ideas you want to generate: ")
    try:
        num_ideas = int(num_input)
    except ValueError:
        print("Invalid input. Defaulting to 1 idea.")
        num_ideas = 1

    for i in range(num_ideas):
        # Generate a new random snippet for each idea
        random_snippet = get_random_snippet(aggregated_text, snippet_length=500)
        prompt = f"Based on my notes: {random_snippet}\nProject Idea:"
        
        idea = generate_project_idea(prompt, model, tokenizer)
        if idea.startswith(prompt):
            idea = idea[len(prompt):].strip()
        
        print(f"\nProject Idea #{i+1}:")
        print(idea)

# ----------------------------
# Main: Orchestrate the Pipeline
# ----------------------------

def main(args):
    # Step 1: Clean and aggregate your notes (reuse your current cleaning pipeline)
    aggregated_file = aggregate_clean_notes(root_dir=args.notes_dir, output_dir=args.cleaned_dir)
    
    # Step 2: Prepare training data (this is a basic example; expand as needed)
    training_file = prepare_training_data(aggregated_file, training_file=args.training_file)
    
    if args.mode == "train":
        # Step 3: Fine-tune the model
        model, tokenizer = fine_tune_model(training_file, model_output_dir=args.model_dir, num_train_epochs=args.epochs)
    else:
        # Load a pre-fine-tuned model (assumes it exists)
        model_name = args.model_dir
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(model_name)
        print(f"Loaded model from {model_name}")

    # Move the model to the appropriate device
    model.to(device)
    
    # Step 4: Launch the CLI interface for inference
    cli_interface(model, tokenizer, aggregated_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLP Pipeline: Clean notes, fine-tune GPT-2, and generate project ideas with a random snippet context.")
    parser.add_argument("--notes_dir", type=str, default="data", help="Directory containing markdown notes.")
    parser.add_argument("--cleaned_dir", type=str, default="cleaned_data", help="Directory to store cleaned notes.")
    parser.add_argument("--training_file", type=str, default="project_ideas_training.txt", help="File to save training data.")
    parser.add_argument("--model_dir", type=str, default="gpt2-finetuned-project-ideas", help="Directory to save/load the fine-tuned model.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train", help="Mode: train a new model or run inference using an existing model.")
    args = parser.parse_args()
    main(args)
