"""
Configuration file - all paths and constants in one place
"""

from pathlib import Path

# ============================================
# REPOSITORY BASE PATH
# ============================================
BASE_DIR = Path(__file__).resolve().parents[1]
BASE_PATH = BASE_DIR

# ============================================
# RAW DATA PATHS (using EXACT folder names from your diagnostic)
# ============================================
RAW_DATA_PATH = BASE_DIR / "data" / "raw"

# Audio paths 
CONTROL_AUDIO_PATH = RAW_DATA_PATH / "Control"
DEMENTIA_AUDIO_PATH = RAW_DATA_PATH / "Dementia"

# Transcript paths
TRANSCRIPTS_PATH = RAW_DATA_PATH / "transcripts"
CONTROL_TRANSCRIPT_PATH = TRANSCRIPTS_PATH / "Control_transcripts"
DEMENTIA_TRANSCRIPT_PATH = TRANSCRIPTS_PATH / "Dementia_transcripts"

# ============================================
# PROCESSED DATA PATHS (where we'll save features)
# ============================================
PROCESSED_PATH = BASE_DIR / "data" / "processed"
ACOUSTIC_FEATURES_PATH = PROCESSED_PATH / "acoustic_features"
LINGUISTIC_FEATURES_PATH = PROCESSED_PATH / "linguistic_features"
METADATA_PATH = PROCESSED_PATH / "metadata"

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
ACOUSTIC_FEATURES_PATH.mkdir(parents=True, exist_ok=True)
LINGUISTIC_FEATURES_PATH.mkdir(parents=True, exist_ok=True)
METADATA_PATH.mkdir(parents=True, exist_ok=True)

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
