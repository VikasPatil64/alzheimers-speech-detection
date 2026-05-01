"""
Multimodal Training with Clinical Features
Audio (WavLM) + Text (BERT) + Clinical (Age, Gender, Education, MMSE)

"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 1. CONFIGURATION

BASE_PATH = r"C:\alzheimers_detection"
METADATA_PATH = os.path.join(BASE_PATH, "data", "processed", "metadata")
EMBEDDINGS_PATH = os.path.join(BASE_PATH, "data", "processed", "embeddings")
ENRICHED_CSV = os.path.join(METADATA_PATH, "enriched_dataset.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 2. LOAD ENRICHED DATASET

print("\n" + "=" * 60)
print("STEP 1: LOADING ENRICHED DATASET")
print("=" * 60)

df = pd.read_csv(ENRICHED_CSV)
y = (df['class'] == 'Dementia').astype(int).values
patient_ids = df['patient_id'].values
print(f"✅ Loaded {len(df)} samples (Control: {sum(y==0)}, Dementia: {sum(y==1)})")


# 3. LOAD EMBEDDINGS

print("\n" + "=" * 60)
print("STEP 2: LOADING EMBEDDINGS")
print("=" * 60)

audio_embeddings = np.load(os.path.join(EMBEDDINGS_PATH, "audio_embeddings.npy"))
text_embeddings = np.load(os.path.join(EMBEDDINGS_PATH, "text_embeddings.npy"))
print(f"✅ Audio: {audio_embeddings.shape}, Text: {text_embeddings.shape}")

# 4. PREPARE CLINICAL FEATURES

print("\n" + "=" * 60)
print("STEP 3: PREPARING CLINICAL FEATURES")
print("=" * 60)

clinical_features = df[['gender', 'education_years', 'age_at_visit', 'mmse_score']].values
imputer = SimpleImputer(strategy='median')
clinical_features = imputer.fit_transform(clinical_features)
scaler = StandardScaler()
clinical_features = scaler.fit_transform(clinical_features)
print(f"✅ Clinical features shape: {clinical_features.shape}")

# ============================================================
# 5. CO-ATTENTION MODEL DEFINITION

class CoAttentionBlock(nn.Module):
    def __init__(self, dim=768, num_heads=8, dropout=0.15):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim*4, dim), nn.Dropout(dropout)
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

class MultimodalModel(nn.Module):
    def __init__(self, audio_dim=512, text_dim=768, clinical_dim=4, proj_dim=768, num_blocks=3, num_heads=8, dropout=0.15):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, proj_dim)
        self.text_proj = nn.Linear(text_dim, proj_dim)
        self.clinical_proj = nn.Sequential(
            nn.Linear(clinical_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, proj_dim)
        )
        self.blocks = nn.ModuleList([CoAttentionBlock(proj_dim, num_heads, dropout) for _ in range(num_blocks)])
        self.fusion = nn.Sequential(
            nn.Linear(proj_dim * 3, proj_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(proj_dim, 2)
        )
    
    def forward(self, audio_feat, text_feat, clinical_feat):
        audio_seq = self.audio_proj(audio_feat).unsqueeze(1)
        text_seq = self.text_proj(text_feat).unsqueeze(1)
        clinical_proj = self.clinical_proj(clinical_feat).unsqueeze(1)
        
        for block in self.blocks:
            audio_seq = block(audio_seq, text_seq)
            text_seq = block(text_seq, audio_seq)
        
        audio_pool = audio_seq.squeeze(1)
        text_pool = text_seq.squeeze(1)
        clinical_pool = clinical_proj.squeeze(1)
        
        concat = torch.cat([audio_pool, text_pool, clinical_pool], dim=1)
        return self.fusion(concat)

# ============================================================
# 6. TRAINING FUNCTION (per fold)
# ============================================================
def train_fold(X_audio, X_text, X_clinical, y_train, X_audio_val, X_text_val, X_clinical_val, y_val):
    model = MultimodalModel().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    audio_train = torch.FloatTensor(X_audio).to(DEVICE)
    text_train = torch.FloatTensor(X_text).to(DEVICE)
    clinical_train = torch.FloatTensor(X_clinical).to(DEVICE)
    y_train_t = torch.LongTensor(y_train).to(DEVICE)
    
    audio_val = torch.FloatTensor(X_audio_val).to(DEVICE)
    text_val = torch.FloatTensor(X_text_val).to(DEVICE)
    clinical_val = torch.FloatTensor(X_clinical_val).to(DEVICE)
    
    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        outputs = model(audio_train, text_train, clinical_train)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_outputs = model(audio_val, text_val, clinical_val)
        val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
        return accuracy_score(y_val, val_preds), f1_score(y_val, val_preds), model

# 7. CROSS-VALIDATION + SAVE BEST MODEL

print("\n" + "=" * 60)
print("STEP 4: SPEAKER-LEVEL CROSS-VALIDATION")
print("=" * 60)

gkf = GroupKFold(n_splits=5)
cv_acc = []
cv_f1 = []
best_accuracy = 0
best_model_state = None

for fold, (train_idx, val_idx) in enumerate(gkf.split(audio_embeddings, y, groups=patient_ids)):
    print(f"\nFold {fold+1}/5")
    
    acc, f1, fold_model = train_fold(
        audio_embeddings[train_idx], text_embeddings[train_idx], clinical_features[train_idx], y[train_idx],
        audio_embeddings[val_idx], text_embeddings[val_idx], clinical_features[val_idx], y[val_idx]
    )
    cv_acc.append(acc)
    cv_f1.append(f1)
    print(f"   Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model_state = fold_model.state_dict()

# 8. SAVE BEST MODEL

print("\n" + "=" * 60)
print("STEP 5: SAVING BEST MODEL")
print("=" * 60)

os.makedirs(os.path.join(BASE_PATH, "models"), exist_ok=True)
model_path = os.path.join(BASE_PATH, "models", "multimodal_clinical_model.pth")
torch.save(best_model_state, model_path)
print(f"✅ Best model saved to: {model_path}")
print(f"   Best validation accuracy: {best_accuracy*100:.2f}%")

# ============================================================
# 9. FINAL RESULTS
# ============================================================
print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"""
┌─────────────────────────────────────────────────────────────┐
│              MULTIMODAL + CLINICAL RESULTS                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  5-fold Cross-Validation:                                   │
│     Mean Accuracy: {np.mean(cv_acc)*100:.1f}%                         │
│     Mean F1-Score:  {np.mean(cv_f1):.3f}                            │
│     Best Accuracy:  {best_accuracy*100:.1f}%                          │
│                                                             │
│  Improvement over baseline (71.3%):                         │
│     +{np.mean(cv_acc)*100 - 71.3:.1f}%                                 │
│                                                             │
│  Model saved to:                                            │
│     {model_path} │
└─────────────────────────────────────────────────────────────┘
""")

print("\n🎉 Training complete! Model saved for Flask app.")

# ============================================================
# SAVE SCALER AND IMPUTER FOR FLASK APP
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: SAVING SCALER AND IMPUTER")
print("=" * 60)

import joblib
scaler_path = os.path.join(BASE_PATH, "models", "clinical_scaler.pkl")
imputer_path = os.path.join(BASE_PATH, "models", "clinical_imputer.pkl")

joblib.dump(scaler, scaler_path)
joblib.dump(imputer, imputer_path)
print(f"✅ Scaler saved to: {scaler_path}")
print(f"✅ Imputer saved to: {imputer_path}")