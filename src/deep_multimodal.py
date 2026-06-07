"""
DEEP MULTIMODAL: WavLM Audio + BERT Text + Neural Network
Expected accuracy: 80-88%
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import WavLMForXVector, Wav2Vec2FeatureExtractor, AutoTokenizer, AutoModel
import librosa
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]
BASE_PATH = BASE_DIR
METADATA_PATH = BASE_DIR / "data" / "processed" / "metadata" / "matched_dataset.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Audio settings
SAMPLE_RATE = 16000
MAX_DURATION = 10  # 10 seconds (shorter = faster, still effective)
MAX_LEN = SAMPLE_RATE * MAX_DURATION

# ============================================================
# 2. LOAD DATA
# ============================================================
print("\n📂 Loading metadata...")
df = pd.read_csv(METADATA_PATH)
df['label'] = (df['class'] == 'Dementia').astype(int)
y = df['label'].values
patient_ids = df['patient_id'].values

print(f"   Samples: {len(df)}")
print(f"   Control: {sum(y==0)}, Dementia: {sum(y==1)}")

# ============================================================
# 3. LOAD MODELS FOR EMBEDDING EXTRACTION
# ============================================================
print("\n🔧 Loading WavLM for audio embeddings...")
audio_processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
audio_model = WavLMForXVector.from_pretrained("microsoft/wavlm-base").to(DEVICE)
audio_model.eval()

print("🔧 Loading BERT for text embeddings...")
text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
text_model = AutoModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
text_model.eval()

# ============================================================
# 4. FUNCTION TO CLEAN TRANSCRIPTS
# ============================================================
import re

def clean_transcript(transcript_path):
    try:
        with open(transcript_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        lines = content.split('\n')
        spoken = []
        for line in lines:
            if line.startswith('*PAR:') or line.startswith('*PAT:') or line.startswith('*CHI:'):
                text = line.split(':', 1)[-1].strip()
                text = re.sub(r'\d+_\d+', '', text)
                text = re.sub(r'\[[^\]]*\]', '', text)
                text = re.sub(r'&-\w+', '', text)
                text = ' '.join(text.split())
                if len(text) > 5:
                    spoken.append(text)
        return ' '.join(spoken)
    except:
        return ""

# ============================================================
# 5. EXTRACT DEEP AUDIO EMBEDDINGS (WavLM)
# ============================================================
print("\n🎵 Extracting deep audio embeddings (WavLM)...")
print("   (This may take 10-15 minutes)")

def get_audio_embedding(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        # Pad or crop
        if len(audio) > MAX_LEN:
            audio = audio[:MAX_LEN]
        else:
            audio = np.pad(audio, (0, MAX_LEN - len(audio)))
        
        inputs = audio_processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = audio_model(**inputs)
        # WavLM-XVector gives 512-dim embeddings
        return outputs.embeddings.cpu().numpy().squeeze()
    except Exception as e:
        print(f"Error: {e}")
        return np.zeros(512)

audio_embeddings = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="   Audio"):
    emb = get_audio_embedding(row['audio_path'])
    audio_embeddings.append(emb)

X_audio = np.array(audio_embeddings)
print(f"   Audio embeddings shape: {X_audio.shape}")

# ============================================================
# 6. EXTRACT DEEP TEXT EMBEDDINGS (BERT)
# ============================================================
print("\n📝 Extracting deep text embeddings (BERT)...")
print("   (This may take 5-10 minutes)")

def get_text_embedding(text):
    if not text.strip():
        return np.zeros(768)
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = text_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()

text_embeddings = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="   Text"):
    text = clean_transcript(row['transcript_path'])
    emb = get_text_embedding(text)
    text_embeddings.append(emb)

X_text = np.array(text_embeddings)
print(f"   Text embeddings shape: {X_text.shape}")

# ============================================================
# 7. COMBINE EMBEDDINGS
# ============================================================
X_multimodal = np.hstack([X_audio, X_text])
print(f"\n🔗 Multimodal features shape: {X_multimodal.shape}")

# ============================================================
# 8. NEURAL NETWORK CLASSIFIER
# ============================================================
class MultimodalNN(nn.Module):
    def __init__(self, input_dim=512+768, hidden_dims=[512, 256, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# ============================================================
# 9. TRAINING FUNCTION
# ============================================================
def train_fold(X_train, y_train, X_val, y_val):
    model = MultimodalNN().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.LongTensor(y_train).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.LongTensor(y_val).to(DEVICE)
    
    # Training loop
    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_preds = torch.argmax(val_outputs, dim=1)
                val_acc = (val_preds == y_val_t).float().mean().item()
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_preds = torch.argmax(val_outputs, dim=1)
        val_acc = accuracy_score(y_val, val_preds.cpu().numpy())
        val_f1 = f1_score(y_val, val_preds.cpu().numpy())
    
    return val_acc, val_f1

# ============================================================
# 10. SPEAKER-LEVEL CROSS-VALIDATION
# ============================================================
print("\n📊 Running speaker-level cross-validation...")

gkf = GroupKFold(n_splits=5)
cv_accuracies = []
cv_f1_scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X_multimodal, y, groups=patient_ids)):
    print(f"\n   Fold {fold+1}/5")
    
    X_train, X_val = X_multimodal[train_idx], X_multimodal[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    acc, f1 = train_fold(X_train, y_train, X_val, y_val)
    cv_accuracies.append(acc)
    cv_f1_scores.append(f1)
    print(f"      Accuracy: {acc:.4f}, F1: {f1:.4f}")

# ============================================================
# 11. FINAL RESULTS
# ============================================================
print("\n" + "=" * 60)
print("FINAL RESULTS - DEEP MULTIMODAL")
print("=" * 60)
print(f"""
┌─────────────────────────────────────────────────────────────┐
│                    DEEP MULTIMODAL RESULTS                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Audio: WavLM deep embeddings (512-dim)                     │
│  Text:  BERT deep embeddings (768-dim)                      │
│  Model: Neural Network (4 layers)                           │
│                                                             │
│  5-fold Cross-Validation:                                   │
│     Mean Accuracy: {np.mean(cv_accuracies)*100:.1f}%                          │
│     Mean F1-Score:  {np.mean(cv_f1_scores)*100:.1f}%                          │
│                                                             │
│  Improvement over BERT-only:                                │
│     +{np.mean(cv_accuracies)*100 - 74.7:.1f}%                               │
└─────────────────────────────────────────────────────────────┘
""")

# ============================================================
# 12. SAVE MODEL
# ============================================================
# Train final model on all data
print("\n💾 Training final model on all data...")
final_model = MultimodalNN().to(DEVICE)
optimizer = optim.AdamW(final_model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

X_all = torch.FloatTensor(X_multimodal).to(DEVICE)
y_all = torch.LongTensor(y).to(DEVICE)

for epoch in range(50):
    optimizer.zero_grad()
    outputs = final_model(X_all)
    loss = criterion(outputs, y_all)
    loss.backward()
    optimizer.step()

# Save
model_path = BASE_DIR / "models" / "deep_multimodal_model.pth"
torch.save(final_model.state_dict(), model_path)
print(f"✅ Model saved to: {model_path}")

print("\n🎉 DEEP MULTIMODAL TRAINING COMPLETE!")
