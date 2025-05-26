# train.py - Dashun Feng
# Trains the TILTRegression model using the tokenized dataset, continuing from checkpoints if applicable

import os
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from model import TILTRegression # Custom TILTRegression Model
from tokenizer import tokenized_dataset # Tokenized article data

output_dir = "./Output"

# Define model: if the model output already exists, continue it. Otherwise, start a new model.
if os.path.exists(os.path.join(output_dir, "pytorch_model.bin")):
    regressionModel = AutoModelForSequenceClassification.from_pretrained(output_dir)
else:
    regressionModel = TILTRegression()

# Define training arguments
training_arguments = TrainingArguments(
    output_dir = "./Output",
    logging_dir = "./output/Logs",
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
    num_train_epochs = 3,
    eval_strategy = "steps",
    eval_steps = 500,
    save_strategy = "steps",
    save_steps = 500,
    save_total_limit = 5,
    logging_steps = 100,
    load_best_model_at_end = True,
    fp16 = True,
    report_to = "tensorboard"
)

# Define trainer
trainer = Trainer(
    model = regressionModel,
    args = training_arguments,
    # Train dataset and eval dataset are the training split (80%) and test split (20%)
    train_dataset = tokenized_dataset['train'], 
    eval_dataset = tokenized_dataset['test'],
)

# Search and define all checkpoints in output directory
print(f"Checking for checkpoints in: {output_dir}")
checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]

# Define checkpoint directory depending on checkpoint search
checkpoint_dir = None
if checkpoints:
    print(f"Found checkpoints: {checkpoints}")
    checkpoint_dir = os.path.join(output_dir, sorted(checkpoints)[-1])  # Use the latest checkpoint
    print(f"Resuming training from checkpoint: {checkpoint_dir}")
else:
    print("No checkpoints found. Starting training from scratch.")

# Begin training (with checkpoint directory if one exists)
trainer.train(resume_from_checkpoint=checkpoint_dir)