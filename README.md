# 🧠 Speech-based Alzheimer’s Disease Detection

A multimodal deep learning system for detecting Alzheimer’s risk using speech, language, and clinical features.

---

## 🚀 Overview

This project analyzes speech recordings to identify early signs of cognitive decline.  
It combines audio, text, and clinical data using a co-attention based neural network.

---

## 🧩 Features

- 🎤 Speech feature extraction using WavLM
- 📝 Text analysis using BERT embeddings
- 🧬 Clinical feature integration (Age, MMSE, Education, Gender)
- 🔁 Co-attention based multimodal fusion
- ⚠️ Confidence-based prediction with uncertainty handling
- 🔍 Speech biomarkers:
  - Pause count
  - Filler word rate
  - Speech rate
  - Pitch variability

---

## 🧠 Model Architecture

- Audio embedding (512-dim) → Projection → 768  
- Text embedding (768-dim)  
- Clinical features → MLP → 768  
- 3 × Co-Attention blocks (Transformer-style)  
- Fusion layer → Binary classification (Dementia / Control)  

---

## 📊 Dataset

- DementiaBank Pitt Corpus (Cookie Theft task)  
- ~439 recordings with transcripts  

---

## ⚙️ Training

- Framework: PyTorch  
- Optimizer: AdamW  
- Loss: CrossEntropyLoss  
- Speaker-independent validation  

---

## 🔍 Prediction Pipeline

1. Audio input  
2. Feature extraction (WavLM)  
3. Text processing (BERT)  
4. Clinical feature input  
5. Multimodal model inference  
6. Confidence-based decision  

---

## 📁 Project Structure

- `app/` – prediction pipeline and UI  
- `src/` – training scripts and models  
- `notebooks/` – experimentation  
- `run.py` – entry point  

---

## ⚠️ Limitations

- Small dataset (~439 samples)  
- Mostly elderly population  
- Not a medical diagnostic tool  

---

## 📌 Future Work

- Model optimization (reduce overfitting)  
- Deployment as a web application  
- Integration of additional biomarkers  

---

## 🛠️ Tech Stack

- Python  
- PyTorch  
- Hugging Face Transformers (WavLM, BERT)  
- NumPy, Pandas  
- Librosa  
- Scikit-learn  
- Flask (for application interface)  
- HTML, CSS, JavaScript  
