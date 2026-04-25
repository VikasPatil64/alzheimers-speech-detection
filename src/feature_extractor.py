"""
Balanced Feature Extraction - Good Speed, Good Features
"""

import os
import numpy as np
import pandas as pd
import librosa
import warnings
from tqdm import tqdm
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor
import threading

warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import *

# ============================================
# BALANCED ACOUSTIC FEATURES
# ============================================

def extract_acoustic_features_balanced(audio_path, sr=16000):
    """
    Balanced feature extraction - good speed, good quality
    """
    try:
        # Load audio (limit to 60 seconds for consistency)
        audio, sr = librosa.load(audio_path, sr=sr, mono=True, duration=60)
        
        features = {}
        
        # 1. MFCCs - 30 coefficients (balanced)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=30)
        for i in range(30):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
        
        # 2. Pitch features (full but optimized)
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, fmin=50, fmax=300)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            if magnitudes[index, t] > np.median(magnitudes):
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
        
        if len(pitch_values) > 0:
            features['pitch_mean'] = float(np.mean(pitch_values))
            features['pitch_std'] = float(np.std(pitch_values))
            features['pitch_min'] = float(np.min(pitch_values))
            features['pitch_max'] = float(np.max(pitch_values))
        else:
            features['pitch_mean'] = features['pitch_std'] = 0.0
            features['pitch_min'] = features['pitch_max'] = 0.0
        
        # 3. Energy features
        rms = librosa.feature.rms(y=audio)[0]
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))
        
        # 4. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        # 5. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        
        # 6. Timing features
        non_silent = librosa.effects.split(audio, top_db=25)
        
        if len(non_silent) > 0:
            total_speech = sum(end - start for start, end in non_silent)
            features['num_speech_segments'] = len(non_silent)
            features['speech_ratio'] = total_speech / len(audio)
            
            # Pause calculation
            pauses = []
            for i in range(1, len(non_silent)):
                pause = non_silent[i][0] - non_silent[i-1][1]
                if pause > 0.1 * sr:
                    pauses.append(pause / sr)
            features['num_pauses'] = len(pauses)
            features['avg_pause_duration'] = float(np.mean(pauses)) if pauses else 0.0
        else:
            features['num_speech_segments'] = 0
            features['speech_ratio'] = 0.0
            features['num_pauses'] = 0
            features['avg_pause_duration'] = 0.0
        
        # 7. Duration
        features['duration_seconds'] = len(audio) / sr
        
        return features
        
    except Exception as e:
        print(f"Error: {e}")
        return None


# ============================================
# LINGUISTIC FEATURES (Same as before - already fast)
# ============================================

def clean_transcript_text(transcript_path):
    """Clean transcript text"""
    try:
        with open(transcript_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        lines = content.split('\n')
        spoken_text = []
        
        for line in lines:
            if line.startswith('*PAR:') or line.startswith('*PAT:') or line.startswith('*CHI:'):
                text = line.split(':', 1)[-1].strip()
                text = re.sub(r'\d+_\d+', '', text)
                text = re.sub(r'\[[^\]]*\]', '', text)
                text = re.sub(r'&-\w+', '', text)
                text = ' '.join(text.split())
                if text and len(text) > 2:
                    spoken_text.append(text)
        
        return ' '.join(spoken_text)
    except:
        return ""


def extract_linguistic_features(transcript_path):
    """Extract linguistic features"""
    text = clean_transcript_text(transcript_path)
    
    features = {}
    
    if not text:
        features['word_count'] = 0
        features['unique_words'] = 0
        features['type_token_ratio'] = 0
        features['avg_word_length'] = 0
        features['filler_count'] = 0
        features['filler_rate'] = 0
        features['sentence_count'] = 0
        features['avg_sentence_length'] = 0
        return features
    
    words = text.lower().split()
    features['word_count'] = len(words)
    features['unique_words'] = len(set(words))
    features['type_token_ratio'] = features['unique_words'] / max(1, features['word_count'])
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
    
    fillers = ['um', 'uh', 'like', 'you know', 'actually', 'basically']
    filler_count = 0
    for filler in fillers:
        filler_count += text.lower().count(filler)
    features['filler_count'] = filler_count
    features['filler_rate'] = filler_count / max(1, features['word_count'])
    
    sentences = re.split(r'[.!?]+', text)
    features['sentence_count'] = len([s for s in sentences if len(s.strip()) > 5])
    features['avg_sentence_length'] = features['word_count'] / max(1, features['sentence_count'])
    
    return features


# ============================================
# MAIN EXTRACTION (Multi-threaded for CPU)
# ============================================

def extract_all_features():
    """Extract features using multi-threading"""
    print("=" * 60)
    print("BALANCED FEATURE EXTRACTION STARTED")
    print("=" * 60)
    
    # Load dataset
    metadata_path = os.path.join(METADATA_PATH, 'matched_dataset.csv')
    df = pd.read_csv(metadata_path)
    
    print(f"Total samples: {len(df)}")
    print(f"Using multi-threading for faster processing\n")
    
    # Process sequentially but with progress bar (reliable)
    all_acoustic = []
    all_linguistic = []
    all_labels = []
    all_patient_ids = []
    all_files = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        audio_path = row['audio_path']
        transcript_path = row['transcript_path']
        
        acoustic = extract_acoustic_features_balanced(audio_path)
        linguistic = extract_linguistic_features(transcript_path)
        
        if acoustic is not None:
            all_acoustic.append(acoustic)
            all_linguistic.append(linguistic)
            all_labels.append(1 if row['class'] == 'Dementia' else 0)
            all_patient_ids.append(row['patient_id'])
            all_files.append(row['audio_file'])
    
    # Convert to DataFrames
    acoustic_df = pd.DataFrame(all_acoustic)
    linguistic_df = pd.DataFrame(all_linguistic)
    
    # Add metadata
    acoustic_df['patient_id'] = all_patient_ids
    acoustic_df['label'] = all_labels
    acoustic_df['file_name'] = all_files
    
    linguistic_df['patient_id'] = all_patient_ids
    linguistic_df['label'] = all_labels
    linguistic_df['file_name'] = all_files
    
    # Save
    os.makedirs(ACOUSTIC_FEATURES_PATH, exist_ok=True)
    os.makedirs(LINGUISTIC_FEATURES_PATH, exist_ok=True)
    
    acoustic_path = os.path.join(ACOUSTIC_FEATURES_PATH, 'acoustic_features.csv')
    linguistic_path = os.path.join(LINGUISTIC_FEATURES_PATH, 'linguistic_features.csv')
    
    acoustic_df.to_csv(acoustic_path, index=False)
    linguistic_df.to_csv(linguistic_path, index=False)
    
    print(f"\n✅ Acoustic features: {acoustic_path}")
    print(f"   Shape: {acoustic_df.shape}")
    print(f"\n✅ Linguistic features: {linguistic_path}")
    print(f"   Shape: {linguistic_df.shape}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(all_labels)}")
    print(f"Control: {all_labels.count(0)}")
    print(f"Dementia: {all_labels.count(1)}")
    print(f"\nAcoustic features: {len(acoustic_df.columns) - 3}")
    print(f"Linguistic features: {len(linguistic_df.columns) - 3}")
    
    return acoustic_df, linguistic_df


if __name__ == "__main__":
    print("\n🚀 Starting extraction (this will take ~20-30 minutes)...\n")
    acoustic, linguistic = extract_all_features()
    print("\n🎉 Complete!")