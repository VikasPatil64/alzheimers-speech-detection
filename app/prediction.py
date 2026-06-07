"""
Multimodal Clinical Model (audio + text + clinical features) - FINAL VERSION
Matches final.ipynb architecture: proj_dim=384, num_blocks=2
"""

import os
from pathlib import Path
import torch
import numpy as np
import librosa
import whisper
from transformers import WavLMForXVector, Wav2Vec2FeatureExtractor, AutoTokenizer, AutoModel
import torch.nn as nn
import warnings
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import parselmouth
import json

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]

# Force offline mode to avoid network issues
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# ============================================================
# Device configuration
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# Model Definitions
# ============================================================

class CoAttentionBlock(nn.Module):
    def __init__(self, dim=384, num_heads=8, dropout=0.25):
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

# Multimodal Clinical Model - MATCHES final.ipynb (proj_dim=384, num_blocks=2)
class MultimodalClinicalModel(nn.Module):
    def __init__(self, audio_dim=512, text_dim=768, clinical_dim=4, proj_dim=384, num_blocks=2, num_heads=8, dropout=0.25):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, proj_dim)
        self.text_proj = nn.Linear(text_dim, proj_dim)
        self.clinical_proj = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, proj_dim)
        )
        self.blocks = nn.ModuleList([CoAttentionBlock(proj_dim, num_heads, dropout) for _ in range(num_blocks)])
        self.fusion = nn.Sequential(
            nn.Linear(proj_dim * 3, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
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
# Global variables
# ============================================================
whisper_model = None
audio_processor = None
audio_model = None
text_tokenizer = None
text_model = None
clinical_model = None
clinical_scaler = None
clinical_imputer = None
optimal_threshold = 0.43  # From final.ipynb cross-validation

# ============================================================
# Helper functions for embeddings
# ============================================================
def get_audio_embedding(audio_array):
    """Extract WavLM embedding from audio numpy array"""
    if isinstance(audio_array, np.ndarray):
        audio = audio_array
    else:
        audio, sr = librosa.load(audio_array, sr=16000, mono=True)
    
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    max_len = 16000 * 60
    if len(audio) > max_len:
        audio = audio[:max_len]
    else:
        audio = np.pad(audio, (0, max_len - len(audio)))
    
    inputs = audio_processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = audio_model(**inputs)
    
    return outputs.embeddings.cpu().numpy().squeeze()

def get_text_embedding(text):
    """Extract BERT embedding from text"""
    if not text or not text.strip():
        return np.zeros(768)
    
    tokens = text_tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = text_model(**tokens)
    
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()

# ============================================================
# DISFLUENCY FEATURE FUNCTIONS
# ============================================================

def extract_pause_features_advanced(audio, sr):
    """Extract pause features that distinguish normal vs dementia"""
    non_silent = librosa.effects.split(audio, top_db=25)
    
    pauses = []
    pause_positions = []
    cumulative_time = 0
    
    for i in range(1, len(non_silent)):
        pause_start = non_silent[i][0] / sr
        pause_end = non_silent[i-1][1] / sr
        pause_duration = pause_start - pause_end
        if pause_duration > 0.15:
            pauses.append(pause_duration)
            pause_positions.append(cumulative_time + pause_start)
    
    speech_durations = [(end - start) / sr for start, end in non_silent]
    
    features = {
        'num_pauses': len(pauses),
        'num_speech_segments': len(non_silent),
        'avg_pause_duration': np.mean(pauses) if pauses else 0,
        'max_pause_duration': np.max(pauses) if pauses else 0,
        'pause_duration_std': np.std(pauses) if pauses else 0,
        'long_pauses_count': sum(1 for p in pauses if p > 2.0),
        'long_pause_total_duration': sum(p for p in pauses if p > 2.0),
        'pauses_per_speech_segment': len(pauses) / max(1, len(non_silent)),
        'speech_segment_duration_std': np.std(speech_durations) if speech_durations else 0,
        'speech_to_pause_ratio': sum(speech_durations) / (sum(pauses) + 0.001),
        'early_pause_ratio': sum(pauses[:max(1, len(pauses)//2)]) / (sum(pauses) + 0.001) if pauses else 0,
    }
    return features

def extract_filler_features(transcript):
    """Extract filler word features"""
    text = transcript.lower()
    
    natural_fillers = ['like', 'you know', 'well', 'so']
    pathological_fillers = ['um', 'uh', 'ah', 'er']
    
    natural_count = sum(text.count(f) for f in natural_fillers)
    pathological_count = sum(text.count(f) for f in pathological_fillers)
    total_words = len(text.split())
    
    return {
        'total_fillers': natural_count + pathological_count,
        'filler_rate': (natural_count + pathological_count) / max(1, total_words),
        'pathological_filler_rate': pathological_count / max(1, total_words),
        'natural_filler_rate': natural_count / max(1, total_words),
        'filler_ratio': pathological_count / max(1, natural_count + 1)
    }

def extract_speech_rate_features(audio, sr, transcript):
    """Extract speaking rate features"""
    word_count = len(transcript.split())
    non_silent = librosa.effects.split(audio, top_db=25)
    speech_duration = sum((end - start) for start, end in non_silent) / sr
    
    return {
        'word_count': word_count,
        'words_per_second': word_count / max(1, speech_duration),
        'silence_ratio': (len(audio)/sr - speech_duration) / max(1, len(audio)/sr)
    }

def extract_pitch_variability(audio, sr):
    """Extract pitch features"""
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, fmin=50, fmax=300)
    
    pitch_values = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        if magnitudes[index, i] > np.median(magnitudes):
            pitch = pitches[index, i]
            if pitch > 0:
                pitch_values.append(pitch)
    
    if len(pitch_values) > 1:
        return {
            'pitch_mean': np.mean(pitch_values),
            'pitch_std': np.std(pitch_values),
            'pitch_range': np.ptp(pitch_values),
            'pitch_variability': np.std(pitch_values) / max(1, np.mean(pitch_values))
        }
    return {'pitch_mean': 0, 'pitch_std': 0, 'pitch_range': 0, 'pitch_variability': 0}

def extract_voice_quality(audio_path):
    """Extract jitter, shimmer using parselmouth"""
    try:
        snd = parselmouth.Sound(audio_path)
        pitch = snd.to_pitch()
        jitter = pitch.get_jitter(local=True)
        
        shimmer = 0
        try:
            point_process = snd.to_point_process("Peaks")
            shimmer = point_process.get_shimmer(local=True)
        except:
            pass
        
        return {'jitter': jitter if jitter else 0, 'shimmer': shimmer if shimmer else 0}
    except:
        return {'jitter': 0, 'shimmer': 0}

# ============================================================
# Load all models
# ============================================================
def load_all_models():
    global whisper_model, audio_processor, audio_model, text_tokenizer, text_model
    global clinical_model, clinical_scaler, clinical_imputer, optimal_threshold
    
    print("Loading Whisper...")
    whisper_model = whisper.load_model("base", device=device)
    
    print("Loading WavLM...")
    audio_processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
    audio_model = WavLMForXVector.from_pretrained("microsoft/wavlm-base").to(device)
    audio_model.eval()
    for param in audio_model.parameters():
        param.requires_grad = False
    
    print("Loading BERT...")
    text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    text_model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
    text_model.eval()
    for param in text_model.parameters():
        param.requires_grad = False
    
    # Load Multimodal Clinical model
    print("Loading Multimodal Clinical model...")
    clinical_model = MultimodalClinicalModel().to(device)
    clinical_model_path = BASE_DIR / "models" / "multimodal_clinical_model.pth"
    
    if clinical_model_path.exists():
        clinical_model.load_state_dict(torch.load(clinical_model_path, map_location=device))
        clinical_model.eval()
        print(f"✅ Model loaded from {clinical_model_path}")
        
        # Load optimal threshold
        threshold_path = BASE_DIR / "models" / "model_info.json"
        if threshold_path.exists():
            with threshold_path.open('r') as f:
                info = json.load(f)
                optimal_threshold = info.get('optimal_threshold', 0.43)
                print(f"✅ Threshold loaded: {optimal_threshold:.3f}")
    else:
        raise FileNotFoundError(f"Model not found at {clinical_model_path}")
    
    # Load clinical scaler and imputer (using _notebook versions)
    scaler_path = BASE_DIR / "models" / "clinical_scaler_notebook.pkl"
    imputer_path = BASE_DIR / "models" / "clinical_imputer_notebook.pkl"
    
    if scaler_path.exists() and imputer_path.exists():
        clinical_scaler = joblib.load(scaler_path)
        clinical_imputer = joblib.load(imputer_path)
        print("✅ Clinical scaler and imputer loaded")
    else:
        print("⚠️ Clinical scaler/imputer not found, creating defaults")
        clinical_scaler = StandardScaler()
        clinical_imputer = SimpleImputer(strategy='median')
    
    print("All models ready!")

# ============================================================
# PREDICTION FUNCTION - KEEP ALL YOUR EXISTING LOGIC
# ============================================================
def predict_from_audio_file_with_clinical(audio_path, clinical_data=None, provided_transcript=None):
    """
    Prediction using Multimodal Clinical Model + Disfluency Features
    """
    print(f"📁 Processing: {audio_path}")
    
    # Parse clinical_data if it's a string
    if clinical_data is None:
        clinical_data = {}
    elif isinstance(clinical_data, str):
        try:
            clinical_data = json.loads(clinical_data)
        except:
            clinical_data = {}
    
    # Check Age (REQUIRED)
    age_raw = clinical_data.get('age_at_visit')
    if age_raw is None or age_raw == '':
        return {
            "prediction": "Incomplete Data",
            "confidence": 0.0,
            "transcript": "",
            "error": "❌ Age is required. Please enter patient age (40-100).",
            "factors": {}
        }
    
    # Check MMSE (REQUIRED)
    mmse_raw = clinical_data.get('mmse_score')
    if mmse_raw is None or mmse_raw == '':
        return {
            "prediction": "Incomplete Data",
            "confidence": 0.0,
            "transcript": "",
            "error": "❌ MMSE score is required. Please enter MMSE (0-30).",
            "factors": {}
        }
    
    # Validate Age range
    try:
        age = int(age_raw)
        if age < 40 or age > 100:
            return {
                "prediction": "Invalid Data",
                "confidence": 0.0,
                "transcript": "",
                "error": f"❌ Age {age} is outside valid range (40-100).",
                "factors": {}
            }
    except:
        return {
            "prediction": "Invalid Data",
            "confidence": 0.0,
            "transcript": "",
            "error": "❌ Invalid age format. Please enter a number.",
            "factors": {}
        }
    
    # Validate MMSE range
    try:
        mmse = int(mmse_raw)
        if mmse < 0 or mmse > 30:
            return {
                "prediction": "Invalid Data",
                "confidence": 0.0,
                "transcript": "",
                "error": f"❌ MMSE score {mmse} is outside valid range (0-30).",
                "factors": {}
            }
    except:
        return {
            "prediction": "Invalid Data",
            "confidence": 0.0,
            "transcript": "",
            "error": "❌ Invalid MMSE format. Please enter a number.",
            "factors": {}
        }
    
    # Gender and education (optional)
    gender = clinical_data.get('gender')
    if gender not in [0, 1, '0', '1']:
        gender = 1
    
    education = clinical_data.get('education_years')
    if education is None or education == '':
        education = 12
    
    clinical_data_clean = {
        'age_at_visit': age,
        'gender': int(gender),
        'education_years': int(education),
        'mmse_score': mmse
    }
    
    print(f"   Clinical data (validated): {clinical_data_clean}")
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    max_len = 16000 * 60
    if len(audio) > max_len:
        audio = audio[:max_len]
    else:
        audio = np.pad(audio, (0, max_len - len(audio)))
    
    # Get transcript
    if provided_transcript and provided_transcript.strip():
        transcript = provided_transcript
    else:
        print("   Transcribing with Whisper...")
        result = whisper_model.transcribe(audio, fp16=False)
        transcript = result["text"]
    
    print(f"   Transcript length: {len(transcript.split())} words")
    
    # Extract disfluency features
    pause_feat = extract_pause_features_advanced(audio, sr)
    
    if pause_feat['long_pauses_count'] >= 2:
        print("   ⚠️ Multiple long pauses (>2 sec) detected – possible word-finding difficulty")
    if pause_feat['pauses_per_speech_segment'] > 1.5:
        print("   ⚠️ Excessive pauses between speech segments")
    
    filler_feat = extract_filler_features(transcript)
    speech_rate_feat = extract_speech_rate_features(audio, sr, transcript)
    pitch_feat = extract_pitch_variability(audio, sr)
    
    word_count = speech_rate_feat['word_count']
    
    # Get embeddings
    audio_emb = get_audio_embedding(audio)
    text_emb = get_text_embedding(transcript)
    
    # Process clinical features
    clinical_values = np.array([[
        clinical_data_clean['gender'],
        clinical_data_clean['education_years'],
        clinical_data_clean['age_at_visit'],
        clinical_data_clean['mmse_score']
    ]])
    
    if clinical_imputer is not None:
        clinical_values = clinical_imputer.transform(clinical_values)
    if clinical_scaler is not None:
        clinical_values = clinical_scaler.transform(clinical_values)
    
    # Predict with clinical model
    audio_t = torch.FloatTensor(audio_emb).unsqueeze(0).to(device)
    text_t = torch.FloatTensor(text_emb).unsqueeze(0).to(device)
    clinical_t = torch.FloatTensor(clinical_values).to(device)
    
    with torch.no_grad():
        logits = clinical_model(audio_t, text_t, clinical_t)
        prob = torch.softmax(logits, dim=1)[0]
        pred = torch.argmax(logits, dim=1).item()
    
    # Override for low speech output
    override = False
    override_reason = ""
    
    if word_count < 50 and pred == 0:
        override = True
        override_reason = f"Very low speech output ({word_count} words in 60 seconds) is a strong indicator of cognitive decline."
        pred = 1
        prob[1] = 0.85
        prob[0] = 0.15
    
    if filler_feat['filler_rate'] > 0.10:
        override = True
        override_reason += f" High filler word rate ({filler_feat['filler_rate']*100:.1f}%) detected."
        prob[1] += 0.05
        prob[1] = min(prob[1], 1.0)
    
    # Safety checks
    age_raw = clinical_data_clean['age_at_visit']
    mmse_raw = clinical_data_clean['mmse_score']
    education_raw = clinical_data_clean['education_years']
    
    duration_sec = len(audio) / sr
    words_per_minute = (word_count / duration_sec) * 60
    
    if age_raw < 60 and mmse_raw > 26 and education_raw > 14:
        print("   ⚠️ SAFETY: Young, educated, high MMSE speaker")
        prob[1] -= 0.15
        override = True
        override_reason = "Young age, high education, and normal MMSE indicate low risk."
    
    if word_count > 120 and words_per_minute > 140:
        print(f"   ⚠️ SAFETY: High word count ({word_count}) and normal speech rate ({words_per_minute:.0f} wpm)")
        prob[1] -= 0.12
        override = True
        override_reason = "Normal speech rate and word count indicate healthy speaker."
    
    if mmse_raw >= 26 and word_count > 80:
        print(f"   ⚠️ SAFETY: Normal MMSE ({mmse_raw}) and adequate word count ({word_count})")
        prob[1] -= 0.10
        override = True
        override_reason = "Normal cognitive score and adequate speech output."
    
    if age_raw < 65 and prob[1] > 0.85:
        prob[1] *= 0.7
        print(f"   ⚠️ Reduced confidence for young speaker (age {age_raw})")
    
    if education_raw > 16 and prob[1] > 0.80:
        prob[1] *= 0.8
        print(f"   ⚠️ Reduced confidence for highly educated speaker ({education_raw} years)")
    
    # Clamp probability
    prob[1] = max(0.0, min(prob[1], 1.0))
    prob[0] = 1 - prob[1]
    
    # Prepare factors for display
    factors = {
        "clinical": {
            "age": clinical_data_clean['age_at_visit'],
            "gender": "Male" if clinical_data_clean['gender'] == 1 else "Female",
            "education": clinical_data_clean['education_years'],
            "mmse": clinical_data_clean['mmse_score']
        },
        "speech": {
            "words_per_minute": f"{speech_rate_feat['words_per_second']*60:.0f}",
            "pause_count": pause_feat['num_pauses'],
            "filler_rate": f"{filler_feat['filler_rate']*100:.1f}%",
            "pitch_variability": f"{pitch_feat['pitch_variability']:.2f}"
        }
    }
    
    # Calculate both probabilities
    dementia_prob = float(prob[1])
    healthy_prob = float(prob[0])

    # Choose the right confidence based on prediction
    if dementia_prob > optimal_threshold:
        final_prediction = "Dementia"
        confidence = dementia_prob      # Show dementia probability
    else:
        final_prediction = "Control"
        confidence = healthy_prob        # Show healthy probability

    return {
        "prediction": final_prediction,
        "confidence": confidence,         # ← Now shows correct probability
        "dementia_probability": dementia_prob,  # Optional: for debugging
        "healthy_probability": healthy_prob,    # Optional: for debugging
        "transcript": transcript,
        "factors": factors,
        "override": override,
        "override_reason": override_reason if override else None
    }


# ============================================================
# Legacy function for backward compatibility
# ============================================================
def predict_audio_file(audio_path):
    """Legacy function - redirects to clinical prediction with defaults"""
    return predict_from_audio_file_with_clinical(audio_path, clinical_data=None, provided_transcript=None)
