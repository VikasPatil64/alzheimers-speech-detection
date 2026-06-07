"""
enrich_dataset.py - Merge clinical metadata with existing dataset
Run this once to create enriched_dataset.csv
"""

import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]
BASE_PATH = BASE_DIR
RAW_DATA_PATH = BASE_DIR / "data" / "raw"
PROCESSED_PATH = BASE_DIR / "data" / "processed"
METADATA_PATH = PROCESSED_PATH / "metadata"

# Ensure metadata directory exists
METADATA_PATH.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. LOAD EXISTING MATCHED DATASET
# ============================================================
print("=" * 60)
print("STEP 1: LOADING EXISTING DATASET")
print("=" * 60)

matched_path = METADATA_PATH / "matched_dataset.csv"
matched_df = pd.read_csv(matched_path)
print(f"✅ Loaded matched dataset: {matched_df.shape}")
print(f"   Columns: {matched_df.columns.tolist()}")

# ============================================================
# 2. LOAD METADATA SPREADSHEET
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: LOADING METADATA SPREADSHEET")
print("=" * 60)

metadata_path = RAW_DATA_PATH / "PItt-data.xlsx"

# Check if file exists
if not metadata_path.exists():
    print(f"❌ ERROR: Metadata file not found at: {metadata_path}")
    print(f"   Please move PItt-data.xlsx to: {RAW_DATA_PATH}")
    exit(1)

# Load the 'data' sheet - header is at row 2 (0-indexed)
metadata_df = pd.read_excel(metadata_path, sheet_name="data", header=2)
print(f"✅ Loaded metadata: {metadata_df.shape}")
print(f"   Columns: {metadata_df.columns.tolist()[:20]}...")

# ============================================================
# 3. EXTRACT RELEVANT CLINICAL FEATURES
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: EXTRACTING CLINICAL FEATURES")
print("=" * 60)

# Select only the columns we need
clinical_cols = [
    'id',           # Patient ID
    'sex',          # Gender (1=male, 0=female)
    'educ',         # Years of education
    'entryage',     # Age at first visit
    'mms',          # MMSE score
    'cdrfs',        # Clinical Dementia Rating
    'basedx'        # Baseline diagnosis
]

# Create a clean clinical dataframe
clinical_df = metadata_df[clinical_cols].copy()

# Rename columns for clarity
clinical_df.columns = [
    'patient_id',
    'gender',
    'education_years',
    'age_at_visit',
    'mmse_score',
    'cdr_score',
    'baseline_dx'
]

# Convert patient_id to string with leading zeros to match matched_dataset
clinical_df['patient_id'] = clinical_df['patient_id'].astype(str).str.zfill(3)

print(f"✅ Clinical features extracted: {clinical_df.shape}")
print(f"   Patient IDs: {clinical_df['patient_id'].nunique()}")
print(clinical_df.head(10))

# ============================================================
# 4. CHECK FOR MISSING VALUES
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: CHECKING MISSING VALUES")
print("=" * 60)

print("Missing values in clinical data:")
for col in clinical_df.columns:
    missing = clinical_df[col].isnull().sum()
    if missing > 0:
        print(f"   {col}: {missing} missing")

# ============================================================
# 5. MERGE WITH MATCHED DATASET
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: MERGING DATASETS")
print("=" * 60)

# Check data types before merge
print(f"matched_df patient_id type: {matched_df['patient_id'].dtype}")
print(f"clinical_df patient_id type: {clinical_df['patient_id'].dtype}")

# Ensure both are strings
matched_df['patient_id'] = matched_df['patient_id'].astype(str).str.zfill(3)

# Merge on patient_id (left join keeps all matched samples)
enriched_df = matched_df.merge(clinical_df, on='patient_id', how='left')

print(f"✅ Enriched dataset shape: {enriched_df.shape}")
print(f"   Samples with clinical data: {enriched_df['mmse_score'].notna().sum()} / {len(enriched_df)}")

# Check which patients are missing clinical data
missing_clinical = enriched_df[enriched_df['mmse_score'].isna()]['patient_id'].unique()
if len(missing_clinical) > 0:
    print(f"\n⚠️ Patients missing clinical data: {missing_clinical.tolist()}")

# ============================================================
# 6. CREATE BINARY DIAGNOSIS FROM BASELINE DX (for verification)
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: DIAGNOSIS MAPPING (Verification)")
print("=" * 60)

def map_dx_to_label(dx):
    if pd.isna(dx):
        return None
    dx = int(dx)
    if dx in [1, 100, 101, 110, 111, 120, 121, 130, 131]:
        return 'Dementia'
    elif dx in [8, 800, 801, 820, 821, 840, 850, 851]:
        return 'Control'
    elif dx in [5, 6, 600, 610, 611, 700, 720, 730, 740]:
        return 'MCI'
    else:
        return 'Other'

clinical_df['dx_label'] = clinical_df['baseline_dx'].apply(map_dx_to_label)
print("Diagnosis distribution from metadata:")
print(clinical_df['dx_label'].value_counts())

# Verify consistency with your labels
print("\nComparing with your dataset labels:")
print(f"Your dataset has: {matched_df['class'].value_counts().to_dict()}")

# ============================================================
# 7. SAVE ENRICHED DATASET
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: SAVING ENRICHED DATASET")
print("=" * 60)

enriched_path = METADATA_PATH / "enriched_dataset.csv"
enriched_df.to_csv(enriched_path, index=False)
print(f"✅ Saved to: {enriched_path}")

# Also save the clinical features separately for reference
clinical_path = METADATA_PATH / "clinical_features.csv"
clinical_df.to_csv(clinical_path, index=False)
print(f"✅ Clinical features saved to: {clinical_path}")

# ============================================================
# 8. DISPLAY SAMPLE OF ENRICHED DATA
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: SAMPLE OF ENRICHED DATA")
print("=" * 60)

# Show first few rows with new columns
display_cols = ['patient_id', 'session', 'class', 'gender', 'education_years', 'age_at_visit', 'mmse_score', 'cdr_score']
print(enriched_df[display_cols].head(10).to_string(index=False))

# ============================================================
# 9. SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("ENRICHMENT SUMMARY")
print("=" * 60)

print(f"""
┌─────────────────────────────────────────────────────────────┐
│                    ENRICHMENT COMPLETE                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Original samples: {len(matched_df)}                                   │
│  Enriched samples: {len(enriched_df)}                                   │
│                                                             │
│  New clinical features added:                               │
│    • gender (0/1) - {enriched_df['gender'].notna().sum()} samples           │
│    • education_years - {enriched_df['education_years'].notna().sum()} samples     │
│    • age_at_visit - {enriched_df['age_at_visit'].notna().sum()} samples         │
│    • mmse_score - {enriched_df['mmse_score'].notna().sum()} samples           │
│    • cdr_score - {enriched_df['cdr_score'].notna().sum()} samples            │
│                                                             │
│  Files created:                                            │
│    • {enriched_path} │
│    • {clinical_path} │
│                                                             │
│  Next steps:                                               │
│    1. Update model to accept clinical features             │
│    2. Retrain multimodal model                            │
│    3. Expected accuracy improvement: +5-10%               │
└─────────────────────────────────────────────────────────────┘
""")

print("\n✅ Done! Ready to update your model.")
