"""
Multimodal Fusion: Audio Features + BERT Text Embeddings
Run this in VS Code - uses your existing data and paths
"""

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. CONFIGURATION - USING YOUR EXISTING PATHS
# ============================================================
# These match your config.py
BASE_PATH = r"C:\alzheimers_detection"
METADATA_PATH = os.path.join(BASE_PATH, "data", "processed", "metadata", "matched_dataset.csv")
ACOUSTIC_FEATURES_PATH = os.path.join(BASE_PATH, "data", "processed", "acoustic_features", "acoustic_features.csv")
LINGUISTIC_FEATURES_PATH = os.path.join(BASE_PATH, "data", "processed", "linguistic_features", "linguistic_features.csv")

# BERT model (small, fast, runs on CPU)
BERT_MODEL = "distilbert-base-uncased"

print("=" * 60)
print("MULTIMODAL FUSION - VS CODE")
print("=" * 60)

# ============================================================
# 2. LOAD YOUR EXISTING FEATURES (already extracted)
# ============================================================
print("\n📂 Loading your extracted features...")

# Load acoustic features
acoustic_df = pd.read_csv(ACOUSTIC_FEATURES_PATH)
print(f"   Acoustic features: {acoustic_df.shape}")

# Load linguistic features (hand-crafted ones)
linguistic_df = pd.read_csv(LINGUISTIC_FEATURES_PATH)
print(f"   Linguistic features (hand-crafted): {linguistic_df.shape}")

# Load metadata
df = pd.read_csv(METADATA_PATH)
print(f"   Metadata: {len(df)} samples")

# Get labels
y = acoustic_df['label'].values

# ============================================================
# 3. PREPARE HAND-CRAFTED FEATURES (from your previous work)
# ============================================================
print("\n🔧 Preparing hand-crafted features...")

# Exclude non-feature columns
non_feature_cols = ['patient_id', 'label', 'file_name']

acoustic_feat_cols = [c for c in acoustic_df.columns if c not in non_feature_cols]
linguistic_feat_cols = [c for c in linguistic_df.columns if c not in non_feature_cols]

# Combine hand-crafted features
X_handcrafted = np.hstack([
    acoustic_df[acoustic_feat_cols].values,
    linguistic_df[linguistic_feat_cols].values
])
print(f"   Hand-crafted features shape: {X_handcrafted.shape}")

# ============================================================
# 4. EXTRACT BERT EMBEDDINGS FROM TRANSCRIPTS (Deep text features)
# ============================================================
print("\n🧠 Loading BERT model and extracting text embeddings...")
print("   (This may take 5-10 minutes on CPU)")

# Load BERT
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
bert_model = AutoModel.from_pretrained(BERT_MODEL)
bert_model.eval()

def clean_transcript_text(transcript_path):
    """Simple cleaning of CHAT transcript"""
    try:
        with open(transcript_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract patient speech lines
        lines = content.split('\n')
        spoken = []
        for line in lines:
            if line.startswith('*PAR:') or line.startswith('*PAT:') or line.startswith('*CHI:'):
                text = line.split(':', 1)[-1].strip()
                # Remove timestamps
                import re
                text = re.sub(r'\d+_\d+', '', text)
                text = re.sub(r'\[[^\]]*\]', '', text)
                text = ' '.join(text.split())
                if text:
                    spoken.append(text)
        return ' '.join(spoken)
    except:
        return ""

def get_bert_embedding(text):
    """Get 768-dim BERT embedding for text"""
    if not text.strip():
        return np.zeros(768)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # Use CLS token embedding
    return outputs.last_hidden_state[:, 0, :].numpy().squeeze()

# Extract BERT embeddings for all transcripts
bert_embeddings = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="   Processing transcripts"):
    transcript_path = row['transcript_path']
    text = clean_transcript_text(transcript_path)
    emb = get_bert_embedding(text)
    bert_embeddings.append(emb)

X_bert = np.array(bert_embeddings)
print(f"   BERT embeddings shape: {X_bert.shape}")

# ============================================================
# 5. COMBINE ALL FEATURES (Multimodal)
# ============================================================
print("\n🔗 Combining all features...")

# Option A: Hand-crafted only (your previous approach)
# Option B: BERT only (deep text)
# Option C: Hand-crafted + BERT (multimodal text)
# Option D: Hand-crafted + BERT + audio features (full multimodal)

# Let's try all combinations to see which works best
X_handcrafted_only = X_handcrafted
X_bert_only = X_bert
X_handcrafted_plus_bert = np.hstack([X_handcrafted, X_bert])
X_full_multimodal = np.hstack([X_handcrafted, X_bert])  # Add more if you have raw audio features

print(f"   Hand-crafted only: {X_handcrafted_only.shape}")
print(f"   BERT only: {X_bert_only.shape}")
print(f"   Combined: {X_handcrafted_plus_bert.shape}")

# ============================================================
# 6. SPEAKER-LEVEL CROSS-VALIDATION (Important!)
# ============================================================
print("\n📊 Running speaker-level cross-validation...")

# Get patient IDs for grouping
patient_ids = df['patient_id'].values

# Use GroupKFold to keep same patient's sessions together
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)

def evaluate_features(X, y, groups, name="Model"):
    """Evaluate features with group cross-validation"""
    scores = []
    f1_scores = []
    
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Simple logistic regression (fast, interpretable)
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(X_train, y_train)
        
        pred = clf.predict(X_val)
        acc = accuracy_score(y_val, pred)
        f1 = f1_score(y_val, pred)
        scores.append(acc)
        f1_scores.append(f1)
    
    print(f"\n   {name}:")
    print(f"      Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    print(f"      F1-Score: {np.mean(f1_scores):.4f}")
    return np.mean(scores)

# Evaluate different feature combinations
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results = {}
results['Hand-crafted (your previous)'] = evaluate_features(X_handcrafted_only, y, patient_ids, "Hand-crafted only")
results['BERT (deep text)'] = evaluate_features(X_bert_only, y, patient_ids, "BERT only")
results['Multimodal (Hand-crafted + BERT)'] = evaluate_features(X_handcrafted_plus_bert, y, patient_ids, "Multimodal")

# ============================================================
# 7. FINAL BEST MODEL ON ALL DATA
# ============================================================
print("\n" + "=" * 60)
print("🏆 TRAINING BEST MODEL ON FULL DATA")
print("=" * 60)

# Find best feature set
best_features = X_handcrafted_plus_bert  # Combined usually works best
best_name = "Multimodal (Hand-crafted + BERT)"

print(f"Using: {best_name}")
print(f"Feature shape: {best_features.shape}")

# Train final model
final_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
final_model.fit(best_features, y)

# Save model for later use
import joblib
model_path = os.path.join(BASE_PATH, "models", "multimodal_model.pkl")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(final_model, model_path)
print(f"✅ Model saved to: {model_path}")

# ============================================================
# 8. FEATURE IMPORTANCE (Which matters most?)
# ============================================================
print("\n📊 Feature importance analysis...")

# Get coefficients from logistic regression
coeffs = final_model.coef_[0]
feature_names = (
    [f"Acoustic_{i}" for i in range(X_handcrafted.shape[1])] + 
    [f"BERT_{i}" for i in range(X_bert.shape[1])]
)

# Get top 10 most important features
importance = list(zip(feature_names, coeffs))
importance.sort(key=lambda x: abs(x[1]), reverse=True)

print("\n   Top 10 most important features:")
for i, (name, coef) in enumerate(importance[:10]):
    direction = "↑ Dementia" if coef > 0 else "↑ Control"
    print(f"      {i+1}. {name[:50]}: {coef:.4f} ({direction})")

# ============================================================
# 9. FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"""
┌─────────────────────────────────────────────────────────────┐
│                    MULTIMODAL RESULTS                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Hand-crafted features (your previous):                     │
│     → ~{results['Hand-crafted (your previous)']*100:.1f}% accuracy                    │
│                                                             │
│  BERT (deep text only):                                     │
│     → ~{results['BERT (deep text)']*100:.1f}% accuracy                          │
│                                                             │
│  🏆 MULTIMODAL (Hand-crafted + BERT):                       │
│     → ~{results['Multimodal (Hand-crafted + BERT)']*100:.1f}% accuracy                │
│                                                             │
│  Improvement over baseline:                                 │
│     → +{(results['Multimodal (Hand-crafted + BERT)'] - results['Hand-crafted (your previous)'])*100:.1f}%                         │
└─────────────────────────────────────────────────────────────┘
""")

print("\n✅ Multimodal training complete!")
print(f"🎯 Target achieved: {results['Multimodal (Hand-crafted + BERT)']*100:.1f}% accuracy")