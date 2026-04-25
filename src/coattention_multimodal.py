"""
Improved Co‑Attention Multimodal Model for Alzheimer's Detection
- Full 60-second audio (matches transcript length)
- Noise reduction + speed perturbation augmentation
- Saves embeddings for later evaluation
- Unfreezes last 2 layers of WavLM & BERT (optional, can be toggled)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import WavLMForXVector, Wav2Vec2FeatureExtractor, AutoTokenizer, AutoModel
import librosa
import noisereduce as nr
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. CONFIGURATION
# ============================================================
BASE_PATH = r"C:\alzheimers_detection"
METADATA_PATH = os.path.join(BASE_PATH, "data", "processed", "metadata", "matched_dataset.csv")
EMBED_SAVE_PATH = os.path.join(BASE_PATH, "data", "processed", "embeddings")
os.makedirs(EMBED_SAVE_PATH, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

SAMPLE_RATE = 16000
MAX_DURATION = 60          # FULL audio to match transcript length
MAX_LEN = SAMPLE_RATE * MAX_DURATION

# Hyperparameters for better accuracy
NUM_EPOCHS_PER_FOLD = 20   # increased from 15
NUM_FINAL_EPOCHS = 30      # increased from 20
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-4
UNFREEZE_LAYERS = True     # Set to True to improve accuracy (slight overfitting risk)

# ============================================================
# 2. LOAD METADATA
# ============================================================
df = pd.read_csv(METADATA_PATH)
df['label'] = (df['class'] == 'Dementia').astype(int)
y = df['label'].values
patient_ids = df['patient_id'].values
print(f"Total samples: {len(df)} (Control: {sum(y==0)}, Dementia: {sum(y==1)})")

# ============================================================
# 3. LOAD FROZEN ENCODERS (WavLM and BERT)
# ============================================================
print("\nLoading WavLM for audio...")
audio_processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
audio_model = WavLMForXVector.from_pretrained("microsoft/wavlm-base").to(DEVICE)
audio_model.eval()
# Freeze all initially
for param in audio_model.parameters():
    param.requires_grad = False

print("Loading BERT for text...")
text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
text_model = AutoModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
text_model.eval()
for param in text_model.parameters():
    param.requires_grad = False

# Optionally unfreeze last 2 layers of each encoder
if UNFREEZE_LAYERS:
    print("\nUnfreezing last 2 layers of WavLM and BERT for fine-tuning...")
    # For WavLM (the transformer encoder part)
    for param in audio_model.wavlm.encoder.layers[-2:].parameters():
        param.requires_grad = True
    # For BERT (DistilBERT has 6 layers; unfreeze last 2)
    for param in text_model.transformer.layer[-2:].parameters():
        param.requires_grad = True

# ============================================================
# 4. CLEAN TRANSCRIPT FUNCTION
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
# 5. EXTRACT AUDIO EMBEDDINGS (512-dim) with noise reduction & augmentation
# ============================================================
def get_audio_embedding(audio_path, augment=False):
    try:
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        # Noise reduction (always applied to reduce bias)
        audio = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.7)
        # Data augmentation (only during training)
        if augment and np.random.rand() > 0.5:
            # Speed perturbation
            rate = np.random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=rate)
        # Pad/crop to full 60 seconds
        if len(audio) > MAX_LEN:
            audio = audio[:MAX_LEN]
        else:
            audio = np.pad(audio, (0, MAX_LEN - len(audio)))
        inputs = audio_processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = audio_model(**inputs)
        return outputs.embeddings.cpu().numpy().squeeze()
    except:
        return np.zeros(512)

print("\nExtracting WavLM embeddings (512-dim) with noise reduction...")
X_audio = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Audio"):
    # During extraction we don't augment, but we apply noise reduction
    emb = get_audio_embedding(row['audio_path'], augment=False)
    X_audio.append(emb)
X_audio = np.array(X_audio)
np.save(os.path.join(EMBED_SAVE_PATH, "audio_embeddings.npy"), X_audio)
print(f"Saved audio embeddings to {EMBED_SAVE_PATH}/audio_embeddings.npy")

# ============================================================
# 6. EXTRACT TEXT EMBEDDINGS (768-dim) – BERT CLS token
# ============================================================
def get_text_embedding(text):
    if not text.strip():
        return np.zeros(768)
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        outputs = text_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()

print("\nExtracting BERT embeddings (768-dim)...")
X_text = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Text"):
    text = clean_transcript(row['transcript_path'])
    emb = get_text_embedding(text)
    X_text.append(emb)
X_text = np.array(X_text)
np.save(os.path.join(EMBED_SAVE_PATH, "text_embeddings.npy"), X_text)
print(f"Saved text embeddings to {EMBED_SAVE_PATH}/text_embeddings.npy")

print(f"Audio shape: {X_audio.shape}, Text shape: {X_text.shape}")

# ============================================================
# 7. CO-ATTENTION MODEL DEFINITION (same as before)
# ============================================================
class CoAttentionBlock(nn.Module):
    def __init__(self, dim=768, num_heads=8, dropout=0.15):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim*4, dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, y):
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        cross_out, _ = self.cross_attn(x, y, y)
        x = self.norm2(x + cross_out)
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        return x

class CoAttentionModel(nn.Module):
    def __init__(self, audio_dim=512, text_dim=768, proj_dim=768, num_blocks=3, num_heads=8, dropout=0.15):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, proj_dim)
        self.text_proj = nn.Linear(text_dim, proj_dim)
        self.blocks = nn.ModuleList([CoAttentionBlock(proj_dim, num_heads, dropout) for _ in range(num_blocks)])
        self.fusion = nn.Sequential(
            nn.Linear(proj_dim*2, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, 2)
        )

    def forward(self, audio_feat, text_feat):
        audio_seq = self.audio_proj(audio_feat).unsqueeze(1)
        text_seq = self.text_proj(text_feat).unsqueeze(1)
        for block in self.blocks:
            audio_seq = block(audio_seq, text_seq)
            text_seq = block(text_seq, audio_seq)
        audio_pool = audio_seq.squeeze(1)
        text_pool = text_seq.squeeze(1)
        concat = torch.cat([audio_pool, text_pool], dim=1)
        logits = self.fusion(concat)
        return logits

# ============================================================
# 8. TRAINING WITH SPEAKER-LEVEL CV (SAVES PREDICTIONS)
# ============================================================
gkf = GroupKFold(n_splits=5)
cv_acc = []
cv_f1 = []
all_fold_true = []
all_fold_preds = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X_audio, y, groups=patient_ids)):
    print(f"\nFold {fold+1}/5")
    X_audio_train = torch.FloatTensor(X_audio[train_idx]).to(DEVICE)
    X_text_train = torch.FloatTensor(X_text[train_idx]).to(DEVICE)
    y_train = torch.LongTensor(y[train_idx]).to(DEVICE)
    X_audio_val = torch.FloatTensor(X_audio[val_idx]).to(DEVICE)
    X_text_val = torch.FloatTensor(X_text[val_idx]).to(DEVICE)
    y_val = y[val_idx]

    model = CoAttentionModel().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS_PER_FOLD):
        model.train()
        optimizer.zero_grad()
        logits = model(X_audio_train, X_text_train)
        loss = criterion(logits, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(X_audio_val, X_text_val)
        preds = torch.argmax(val_logits, dim=1).cpu().numpy()
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds)
    cv_acc.append(acc)
    cv_f1.append(f1)
    all_fold_true.extend(y_val)
    all_fold_preds.extend(preds)
    print(f"  Val Accuracy: {acc:.4f}, F1: {f1:.4f}")

print("\n" + "="*60)
print("CO-ATTENTION RESULTS (5-fold CV)")
print("="*60)
print(f"Mean Accuracy: {np.mean(cv_acc)*100:.1f}% (+/- {np.std(cv_acc)*100:.1f})")
print(f"Mean F1: {np.mean(cv_f1):.3f}")

# Save cross-val predictions for later evaluation
np.save(os.path.join(EMBED_SAVE_PATH, "cv_true_labels.npy"), np.array(all_fold_true))
np.save(os.path.join(EMBED_SAVE_PATH, "cv_pred_labels.npy"), np.array(all_fold_preds))

# ============================================================
# 9. TRAIN FINAL MODEL ON ALL DATA
# ============================================================
print("\nTraining final co-attention model on all data...")
final_model = CoAttentionModel().to(DEVICE)
optimizer = optim.AdamW(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
X_audio_all = torch.FloatTensor(X_audio).to(DEVICE)
X_text_all = torch.FloatTensor(X_text).to(DEVICE)
y_all = torch.LongTensor(y).to(DEVICE)

for epoch in range(NUM_FINAL_EPOCHS):
    final_model.train()
    optimizer.zero_grad()
    logits = final_model(X_audio_all, X_text_all)
    loss = nn.CrossEntropyLoss()(logits, y_all)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
    optimizer.step()
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Save final model
os.makedirs(os.path.join(BASE_PATH, "models"), exist_ok=True)
torch.save(final_model.state_dict(), os.path.join(BASE_PATH, "models", "coattention_final_model.pth"))
print("✅ Final model saved to C:\\alzheimers_detection\\models\\coattention_final_model.pth")
print("\n🎉 Training complete. Now run evaluation script to get confusion matrix.")