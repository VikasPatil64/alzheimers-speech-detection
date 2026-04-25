"""
Wav2Vec2 Training - Completely avoids safetensors by using PyTorch-only loading
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================
# FIXED LENGTH AUDIO DATASET
# ============================================
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, processor, target_sr=16000, fixed_length=960000):
        self.file_paths = file_paths
        self.labels = labels
        self.processor = processor
        self.target_sr = target_sr
        self.fixed_length = fixed_length
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        audio, _ = librosa.load(self.file_paths[idx], sr=self.target_sr)
        
        if len(audio) < self.fixed_length:
            audio = np.pad(audio, (0, self.fixed_length - len(audio)))
        else:
            audio = audio[:self.fixed_length]
        
        inputs = self.processor(audio, sampling_rate=self.target_sr, return_tensors="pt")
        
        return {
            'input_values': inputs.input_values[0],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ============================================
# CUSTOM MODEL (No safetensors involved)
# ============================================
class Wav2Vec2CustomClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Load only the base model (not the classification wrapper)
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        # Add custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, input_values):
        # Get embeddings from Wav2Vec2
        outputs = self.wav2vec2(input_values)
        # Take mean over time dimension
        embeddings = outputs.last_hidden_state.mean(dim=1)
        # Classify
        logits = self.classifier(embeddings)
        return logits

# ============================================
# DATA COLLATOR
# ============================================
class DataCollator:
    def __call__(self, features):
        input_values = [f["input_values"] for f in features]
        labels = [f["labels"] for f in features]
        
        # Pad to same length
        max_len = max(len(x) for x in input_values)
        padded = []
        for x in input_values:
            if len(x) < max_len:
                pad = torch.zeros(max_len - len(x))
                padded.append(torch.cat([x, pad]))
            else:
                padded.append(x)
        
        return {
            'input_values': torch.stack(padded),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

# ============================================
# MAIN
# ============================================
print("="*60)
print("LOADING DATASET")
print("="*60)

metadata_path = r"C:\alzheimers_detection\data\processed\metadata\matched_dataset.csv"
df = pd.read_csv(metadata_path)
df['label'] = (df['class'] == 'Dementia').astype(int)

file_paths = df['audio_path'].tolist()
labels = df['label'].tolist()

print(f"✅ Loaded {len(file_paths)} samples (Control: {labels.count(0)}, Dementia: {labels.count(1)})")

train_paths, test_paths, train_labels, test_labels = train_test_split(
    file_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

print(f"✅ Train: {len(train_paths)}, Test: {len(test_paths)}")

# ============================================
# LOAD PROCESSOR
# ============================================
print("\n"+"="*60)
print("LOADING PROCESSOR")
print("="*60)

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
print("✅ Processor loaded")

# ============================================
# CREATE DATASETS
# ============================================
print("\n"+"="*60)
print("CREATING DATASETS")
print("="*60)

train_dataset = AudioDataset(train_paths, train_labels, processor)
test_dataset = AudioDataset(test_paths, test_labels, processor)
data_collator = DataCollator()

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=data_collator)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=data_collator)
print("✅ DataLoaders created")

# ============================================
# LOAD MODEL (NO SAFETENSORS)
# ============================================
print("\n"+"="*60)
print("LOADING MODEL (PyTorch only - no safetensors)")
print("="*60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Wav2Vec2CustomClassifier().to(device)
print(f"✅ Model loaded on {device}")

# ============================================
# TRAINING
# ============================================
print("\n"+"="*60)
print("STARTING TRAINING")
print("="*60)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
criterion = nn.CrossEntropyLoss()

best_accuracy = 0
num_epochs = 10

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
        input_values = batch['input_values'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_values)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
            input_values = batch['input_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_values)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    print(f"\n📊 Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Acc={accuracy*100:.2f}%, F1={f1:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), r"C:\alzheimers_detection\models\best_model.pth")
        print(f"   ✅ Best model saved! (Acc: {accuracy*100:.2f}%)")

# ============================================
# FINAL RESULTS
# ============================================
print("\n"+"="*60)
print("FINAL RESULTS")
print("="*60)
print(f"🏆 Best Accuracy: {best_accuracy*100:.2f}%")
print(f"✅ Model saved to: C:\\alzheimers_detection\\models\\best_model.pth")
print("="*60)