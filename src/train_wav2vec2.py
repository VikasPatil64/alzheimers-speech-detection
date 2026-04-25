# src/train_wav2vec2.py

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datasets import Dataset, Audio
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. CONFIGURATION & DATA LOADING
# ============================================
print("="*60)
print("STEP 1: LOADING DATASET")
print("="*60)

# Load the metadata
metadata_path = r"C:\alzheimers_detection\data\processed\metadata\matched_dataset.csv"
df = pd.read_csv(metadata_path)


# Create label column (1 for Dementia, 0 for Control)
df['label'] = (df['class'] == 'Dementia').astype(int)

# Prepare data: 'audio_path' and 'label'
df = df[['audio_path', 'label']]
print(f"✅ Loaded {len(df)} samples.")
# Split into training and testing sets
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df['label'], random_state=42
)
print(f"✅ Train set: {len(train_df)} samples")
print(f"✅ Test set: {len(test_df)} samples")

# ============================================
# 2. LOAD PROCESSOR & PREPARE DATASET
# ============================================
print("\n"+"="*60)
print("STEP 2: LOADING WAV2VEC2 PROCESSOR")
print("="*60)

# The processor handles both feature extraction and tokenization
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
print("✅ Wav2Vec2 processor loaded.")

# Convert pandas DataFrames to Hugging Face Dataset objects
train_dataset = Dataset.from_pandas(train_df[['audio_path', 'label']])
test_dataset = Dataset.from_pandas(test_df[['audio_path', 'label']])

# Cast the 'audio_path' column to the 'Audio' type. 
# This lets the dataset object load the audio file on the fly.
train_dataset = train_dataset.cast_column("audio_path", Audio(sampling_rate=16000))
test_dataset = test_dataset.cast_column("audio_path", Audio(sampling_rate=16000))

def preprocess_function(examples):
    """
    This function loads the audio and prepares it for the model.
    """
    # The dataset automatically loads the audio when we access the column.
    audio_arrays = [x["array"] for x in examples["audio_path"]]
    # Process the raw audio arrays into the model's input format.
    inputs = processor(audio_arrays, sampling_rate=16000, padding=True, return_tensors="pt")
    return inputs

# Apply the preprocessing function to the datasets
print("\n🔄 Preprocessing training dataset...")
train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["audio_path"])
print("🔄 Preprocessing test dataset...")
test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=["audio_path"])
print("✅ Preprocessing complete.")

# Rename 'label' to 'labels' as expected by the Hugging Face Trainer
train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

# Set the format for PyTorch tensors
train_dataset.set_format(type='torch', columns=['input_values', 'labels'])
test_dataset.set_format(type='torch', columns=['input_values', 'labels'])

# ============================================
# 3. LOAD MODEL AND DEFINE METRICS
# ============================================
print("\n"+"="*60)
print("STEP 3: LOADING WAV2VEC2 MODEL")
print("="*60)

# Load the pre-trained model for sequence classification
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=2,  # We have two classes: Control (0) and Dementia (1)
    ignore_mismatched_sizes=True
)
print("✅ Wav2Vec2 model loaded and classification head added.")

def compute_metrics(eval_pred):
    """Calculate accuracy, precision, recall, and f1-score."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# ============================================
# 4. TRAINING CONFIGURATION AND EXECUTION
# ============================================
print("\n"+"="*60)
print("STEP 4: SETTING UP TRAINING")
print("="*60)

# Training arguments are key to a successful fine-tuning
training_args = TrainingArguments(
    output_dir="./wav2vec2-dementia-results",  # Where to save model checkpoints
    evaluation_strategy="epoch",               # Evaluate at the end of each epoch
    save_strategy="epoch",                     # Save model at the end of each epoch
    learning_rate=3e-5,                        # A good default learning rate for fine-tuning
    per_device_train_batch_size=8,             # Batch size for training. Lower if you run out of memory.
    per_device_eval_batch_size=8,              # Batch size for evaluation.
    num_train_epochs=10,                       # Number of training epochs
    weight_decay=0.01,                         # Regularization to prevent overfitting
    warmup_ratio=0.1,                          # Proportion of training steps to warm up the learning rate
    logging_dir='./logs',                      # Directory for logs
    logging_steps=10,                          # Log every 10 steps
    load_best_model_at_end=True,               # Load the best model when finished
    metric_for_best_model="accuracy",          # Use accuracy to determine the best model
    fp16=torch.cuda.is_available(),            # Use mixed precision if a GPU is available
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=processor.feature_extractor,     # Use the processor for feature extraction
    compute_metrics=compute_metrics,
)

print("✅ Training setup complete. Starting training...")
print("⏳ This will take ~20-30 minutes on a CPU, or less if you have a GPU.\n")

# ============================================
# 5. TRAIN AND SAVE THE MODEL
# ============================================
trainer.train()

print("\n"+"="*60)
print("STEP 5: EVALUATING AND SAVING THE MODEL")
print("="*60)

# Evaluate the final model on the test set
eval_results = trainer.evaluate()
print(f"\n🏆 Final Test Set Results:")
print(f"   Accuracy:  {eval_results['eval_accuracy']:.4f} ({eval_results['eval_accuracy']*100:.2f}%)")
print(f"   Precision: {eval_results['eval_precision']:.4f}")
print(f"   Recall:    {eval_results['eval_recall']:.4f}")
print(f"   F1-Score:  {eval_results['eval_f1']:.4f}")

# Save the fine-tuned model and processor
model_save_path = r"C:\alzheimers_detection\models\wav2vec2-dementia-model"
processor.save_pretrained(model_save_path)
model.save_pretrained(model_save_path)
print(f"\n✅ Model and processor saved to: {model_save_path}")

print("\n"+"="*60)
print("🎉 TRAINING COMPLETE!")
print("="*60)