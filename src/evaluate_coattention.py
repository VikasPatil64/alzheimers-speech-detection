"""
Complete Evaluation for Co-Attention Model
Generates: Confusion Matrix, Precision, Recall, F1, Specificity, Per-class metrics
"""

import os
import sys
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import GroupKFold

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import from the same directory
from coattention_multimodal import CoAttentionModel

import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD DATA
# ============================================================
BASE_PATH = r"C:\alzheimers_detection"
METADATA_PATH = os.path.join(BASE_PATH, "data", "processed", "metadata", "matched_dataset.csv")

df = pd.read_csv(METADATA_PATH)
df['label'] = (df['class'] == 'Dementia').astype(int)
y = df['label'].values
patient_ids = df['patient_id'].values

print("=" * 60)
print("CO-ATTENTION MODEL EVALUATION")
print("=" * 60)
print(f"Total samples: {len(df)} (Control: {sum(y==0)}, Dementia: {sum(y==1)})")

# ============================================================
# 2. LOAD EMBEDDINGS
# ============================================================
embeddings_path = os.path.join(BASE_PATH, "data", "processed", "embeddings")
os.makedirs(embeddings_path, exist_ok=True)

audio_emb_path = os.path.join(embeddings_path, "audio_embeddings.npy")
text_emb_path = os.path.join(embeddings_path, "text_embeddings.npy")

# Check if embeddings are saved from training
if os.path.exists(audio_emb_path) and os.path.exists(text_emb_path):
    print("\n📂 Loading saved embeddings...")
    X_audio = np.load(audio_emb_path)
    X_text = np.load(text_emb_path)
    print(f"   Audio shape: {X_audio.shape}, Text shape: {X_text.shape}")
else:
    print("\n⚠️ Embeddings not found. Re-extracting from coattention script...")
    print("   Please run coattention_multimodal.py first to generate embeddings.")
    print("   Or wait - the embeddings should be generated during training.")
    exit()

# ============================================================
# 3. LOAD MODEL
# ============================================================
print("\n🔧 Loading trained Co-Attention model...")
device = torch.device("cpu")
model = CoAttentionModel().to(device)
model_path = os.path.join(BASE_PATH, "models", "coattention_model.pth")

if not os.path.exists(model_path):
    print(f"❌ Model not found at {model_path}")
    print("   Please run coattention_multimodal.py first to train the model.")
    exit()

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ Model loaded")

# ============================================================
# 4. CROSS-VALIDATION PREDICTIONS (Using same folds as training)
# ============================================================
print("\n📊 Running evaluation on cross-validation folds...")

gkf = GroupKFold(n_splits=5)
all_true = []
all_preds = []
all_probs = []
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X_audio, y, groups=patient_ids)):
    print(f"\n   Fold {fold+1}/5")
    
    X_audio_val = torch.FloatTensor(X_audio[val_idx]).to(device)
    X_text_val = torch.FloatTensor(X_text[val_idx]).to(device)
    y_val = y[val_idx]
    
    with torch.no_grad():
        logits = model(X_audio_val, X_text_val)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    
    all_true.extend(y_val)
    all_preds.extend(preds)
    all_probs.extend(probs[:, 1].cpu().numpy())
    
    acc = accuracy_score(y_val, preds)
    fold_accuracies.append(acc)
    print(f"      Accuracy: {acc:.4f} ({acc*100:.2f}%)")

# ============================================================
# 5. GENERATE ALL METRICS
# ============================================================
print("\n" + "=" * 60)
print("FINAL EVALUATION RESULTS")
print("=" * 60)

# Overall metrics
accuracy = accuracy_score(all_true, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_true, all_preds, average='binary')

print(f"\n📊 Cross-Validation Summary:")
print(f"   Fold accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")
print(f"   Mean Accuracy:   {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")

print(f"\n📊 Overall Metrics (across all folds):")
print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")

# Per-class metrics
print(f"\n📊 Per-Class Classification Report:")
print(classification_report(all_true, all_preds, target_names=['Control (0)', 'Dementia (1)']))

# Confusion Matrix
cm = confusion_matrix(all_true, all_preds)
tn, fp, fn, tp = cm.ravel()

print(f"\n📊 Confusion Matrix:")
print(f"                 Predicted")
print(f"                 Control  Dementia")
print(f"Actual Control   {tn:7d}  {fp:7d}")
print(f"       Dementia   {fn:7d}  {tp:7d}")

# Derived metrics
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0
balanced_acc = (sensitivity + specificity) / 2

print(f"\n📊 Detailed Clinical Metrics:")
print(f"   Sensitivity (Recall - Dementia):      {sensitivity:.4f}")
print(f"   Specificity (Recall - Control):       {specificity:.4f}")
print(f"   Balanced Accuracy:                    {balanced_acc:.4f}")
print(f"   PPV (Precision - Dementia):           {ppv:.4f}")
print(f"   NPV (Precision - Control):            {npv:.4f}")

# Save results to file
results_path = os.path.join(BASE_PATH, "models", "evaluation_results.txt")
with open(results_path, 'w') as f:
    f.write("CO-ATTENTION MODEL EVALUATION RESULTS\n")
    f.write("=" * 50 + "\n")
    f.write(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})\n")
    f.write(f"Overall Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n")
    f.write(f"Sensitivity: {sensitivity:.4f}\n")
    f.write(f"Specificity: {specificity:.4f}\n")
    f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n")
    f.write("\nConfusion Matrix:\n")
    f.write(f"[[{tn}, {fp}],\n")
    f.write(f" [{fn}, {tp}]]\n")

print(f"\n💾 Results saved to: {results_path}")

print("\n" + "=" * 60)
print("✅ Evaluation Complete!")
print("=" * 60)