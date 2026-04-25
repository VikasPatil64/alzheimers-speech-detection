"""
Configuration file - all paths and constants in one place
"""

import os

# ============================================
# YOUR BASE PATH - CONFIRMED WORKING
# ============================================
BASE_PATH = r"C:\alzheimers_detection"

# ============================================
# RAW DATA PATHS (using EXACT folder names from your diagnostic)
# ============================================
RAW_DATA_PATH = os.path.join(BASE_PATH, "data", "raw")

# Audio paths 
CONTROL_AUDIO_PATH = os.path.join(RAW_DATA_PATH, "Control")     
DEMENTIA_AUDIO_PATH = os.path.join(RAW_DATA_PATH, "Dementia")   

# Transcript paths
TRANSCRIPTS_PATH = os.path.join(RAW_DATA_PATH, "transcripts")
CONTROL_TRANSCRIPT_PATH = os.path.join(TRANSCRIPTS_PATH, "Control_transcripts")
DEMENTIA_TRANSCRIPT_PATH = os.path.join(TRANSCRIPTS_PATH, "Dementia_transcripts")

# ============================================
# PROCESSED DATA PATHS (where we'll save features)
# ============================================
PROCESSED_PATH = os.path.join(BASE_PATH, "data", "processed")
ACOUSTIC_FEATURES_PATH = os.path.join(PROCESSED_PATH, "acoustic_features")
LINGUISTIC_FEATURES_PATH = os.path.join(PROCESSED_PATH, "linguistic_features")
METADATA_PATH = os.path.join(PROCESSED_PATH, "metadata")

# ============================================
# AUDIO PROCESSING CONSTANTS
# ============================================
SAMPLE_RATE = 16000  # Target sample rate (16kHz)
MAX_AUDIO_DURATION = 120  # Max seconds per audio file
MIN_AUDIO_DURATION = 10   # Min seconds per audio file

# ============================================
# FEATURE EXTRACTION CONSTANTS
# ============================================
N_MFCC = 40  # Number of MFCC coefficients
FRAME_SIZE = 0.025  # 25ms frame size
HOP_SIZE = 0.010    # 10ms hop size

# ============================================
# CREATE DIRECTORIES (don't change this)
# ============================================
os.makedirs(ACOUSTIC_FEATURES_PATH, exist_ok=True)
os.makedirs(LINGUISTIC_FEATURES_PATH, exist_ok=True)
os.makedirs(METADATA_PATH, exist_ok=True)

print("=" * 50)
print("CONFIGURATION LOADED SUCCESSFULLY!")
print("=" * 50)
print(f"Base Path: {BASE_PATH}")
print(f"\nAudio paths:")
print(f"  Control: {CONTROL_AUDIO_PATH}")
print(f"  Dementia: {DEMENTIA_AUDIO_PATH}")
print(f"\nTranscript paths:")
print(f"  Control: {CONTROL_TRANSCRIPT_PATH}")
print(f"  Dementia: {DEMENTIA_TRANSCRIPT_PATH}")
print(f"\nProcessed Path: {PROCESSED_PATH}")
print("=" * 50)