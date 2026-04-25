import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

BASE_PATH = r"C:\alzheimers_detection"
EMBED_PATH = os.path.join(BASE_PATH, "data", "processed", "embeddings")

# Load saved predictions from cross-validation
true_labels = np.load(os.path.join(EMBED_PATH, "cv_true_labels.npy"))
pred_labels = np.load(os.path.join(EMBED_PATH, "cv_pred_labels.npy"))

# Metrics
acc = accuracy_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)
cm = confusion_matrix(true_labels, pred_labels)

print("="*60)
print("FINAL EVALUATION (Cross-Validation Predictions)")
print("="*60)
print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=['Control', 'Dementia']))
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Control','Dementia'], yticklabels=['Control','Dementia'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Co-Attention Model')
plt.tight_layout()
plt.savefig(os.path.join(BASE_PATH, "models", "confusion_matrix.png"))
print("\n✅ Confusion matrix saved to models/confusion_matrix.png")