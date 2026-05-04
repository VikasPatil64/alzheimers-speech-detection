# 🧠 Speech-based Alzheimer’s Disease Detection

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-red)
![Accuracy](https://img.shields.io/badge/Accuracy-89.29%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Research%20Only-yellow)

A multimodal deep learning system for detecting Alzheimer’s risk using speech, language, and clinical features.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Best Validation Accuracy** | **89.29%** |
| **5-Fold Cross-Validation Accuracy** | **83.37% ± 1.55%** |
| **AUC-ROC** | **0.87** |
| **Precision** | 0.78 |
| **Recall** | 0.82 |
| **F1-Score** | 0.80 |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- 8GB+ RAM (16GB recommended)

### Installation

```bash
# Clone repository
git clone https://github.com/VikasPatil64/alzheimers-speech-detection.git
cd alzheimers-speech-detection

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python run.py

## 🧩 Features

- 🎤 **Speech feature extraction** using WavLM (512-dim embeddings)
- 📝 **Text analysis** using BERT embeddings (768-dim)
- 🧬 **Clinical feature integration** (Age, MMSE, Education, Gender)
- 🔁 **Co-attention based multimodal fusion**
- ⚠️ **Confidence-based prediction** with optimal threshold (0.43)
- 🔍 **Speech biomarkers**:
  - Pause count and duration analysis
  - Filler word rate (um, uh, like, you know)
  - Speech rate (words per minute)
  - Pitch variability
  - Long pause detection (>2 seconds)

Key Parameters:

Audio embedding dimension: 512 → 384

Text embedding dimension: 768 → 384

Clinical features: 4 → 64 → 384

Co-Attention blocks: 2

Attention heads: 8

Dropout: 0.15

Total trainable parameters: 5,694,274

📁 Project Structure

alzheimers-speech-detection/
├── app/                      # Flask web application
│   ├── app.py               # Main Flask routes
│   ├── prediction.py        # Model inference pipeline
│   ├── static/              # CSS, JS, images
│   └── templates/           # HTML templates
├── data/                    # Dataset (not included in repo)
│   ├── raw/                 # Original audio/transcripts
│   └── processed/           # Precomputed embeddings
├── models/                  # Saved model weights
│   ├── multimodal_clinical_model.pth
│   ├── clinical_scaler_notebook.pkl
│   ├── clinical_imputer_notebook.pkl
│   └── model_info.json
├── notebooks/               # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 03_finetuning_experiment.ipynb
│   └── final.ipynb          # Final training notebook
├── src/                     # Source modules
│   ├── audio_preprocessor.py
│   ├── config.py
│   └── enrich_dataset.py
├── requirements.txt         # Python dependencies
├── run.py                   # Application entry point
└── README.md                # This file


🔍 Prediction Pipeline
Audio Input (live recording or file upload)

Feature Extraction

Audio → WavLM → 512-dim embedding

Speech → Whisper transcription → BERT → 768-dim embedding

Clinical features (Age, Gender, Education, MMSE)

Multimodal Inference

Co-attention fusion

Softmax probability

Confidence-based Decision

Optimal threshold: 0.43 (from cross-validation)

Output: Dementia / Control with confidence score

Speech Biomarkers Display

Pause count, filler rate, speech rate, pitch variability


📊 Dataset
Source: DementiaBank Pitt Corpus (Cookie Theft picture description task)

Total samples: 439

Control (Healthy): 217 samples (94 patients)

Dementia: 222 samples (124 patients)

Audio duration: 60 seconds (standardized)

Format: MP3/WAV audio + CHAT transcripts

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Deep Learning** | PyTorch 2.11 |
| **Audio Model** | Microsoft WavLM (Hugging Face) |
| **Text Model** | DistilBERT (Hugging Face) |
| **Transcription** | OpenAI Whisper (base) |
| **Web Framework** | Flask |
| **Audio Processing** | Librosa, Noisereduce |
| **Data Processing** | NumPy, Pandas, Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |


⚠️ Limitations
Small dataset (~439 samples) - limited generalization

Single dataset source (Pitt Corpus) - may not generalize to other populations

Binary classification only - cannot detect MCI (Mild Cognitive Impairment)

Limited clinical features (only 4) - missing genetics, family history, neuroimaging

English only - not tested for other languages


⚠️ Medical Disclaimer
This tool is for RESEARCH and EDUCATIONAL purposes only.

It is NOT FDA approved and NOT intended for clinical diagnosis.

Decisions about Alzheimer's disease should ALWAYS be made by qualified medical professionals using standard clinical assessments.

This system may produce incorrect predictions. Do NOT use it to make healthcare decisions.

📌 Future Work
Expand dataset (external validation, more samples)

Add MCI (3-class classification)

Real-time streaming inference

Mobile app deployment

Add more clinical features (family history, APOE-e4 status)

Multi-language support

Longitudinal patient tracking

Explainability module (LIME/SHAP)

Cloud deployment (AWS/GCP)

📄 License
MIT License - See LICENSE file for details.

🙏 Acknowledgments
DementiaBank for the Pitt Corpus dataset

Hugging Face for transformer models

OpenAI for Whisper transcription

📧 Contact
For questions or collaboration opportunities, please open an issue on GitHub.

⭐ If you find this project useful, please consider giving it a star!

text

## Steps:

1. **Select all** the text above (Ctrl+A)
2. **Copy** it (Ctrl+C)
3. **Open** your `README.md` file in VS Code or any text editor
4. **Select all** existing content (Ctrl+A)
5. **Delete** it (Delete key)
6. **Paste** the new content (Ctrl+V)
7. **Save** the file (Ctrl+S)

Then commit:

```bash
git add README.md
git commit -m "Update README with correct architecture and complete results"
git push origin main
