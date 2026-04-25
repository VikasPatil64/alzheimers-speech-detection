# # # verify_features.py
# # import pandas as pd
# # import os

# # # Load features
# # acoustic_path = r"C:\alzheimers_detection\data\processed\acoustic_features\acoustic_features.csv"
# # linguistic_path = r"C:\alzheimers_detection\data\processed\linguistic_features\linguistic_features.csv"

# # acoustic_df = pd.read_csv(acoustic_path)
# # linguistic_df = pd.read_csv(linguistic_path)

# # print("=" * 50)
# # print("FEATURES LOADED SUCCESSFULLY")
# # print("=" * 50)
# # print(f"\nAcoustic Features: {acoustic_df.shape}")
# # print(f"  Columns: {list(acoustic_df.columns[:10])}...")
# # print(f"  Label distribution: {acoustic_df['label'].value_counts().to_dict()}")

# # print(f"\nLinguistic Features: {linguistic_df.shape}")
# # print(f"  Columns: {list(linguistic_df.columns)}")
# # print(f"  Label distribution: {linguistic_df['label'].value_counts().to_dict()}")

# # # Check for missing values
# # print(f"\nMissing values in acoustic features: {acoustic_df.isnull().sum().sum()}")
# # print(f"Missing values in linguistic features: {linguistic_df.isnull().sum().sum()}")

# # print("\n✅ Ready for model training!")

# import whisper
# import time

# start = time.time()
# model = whisper.load_model("base")
# print(f"Loaded in {time.time() - start:.2f} seconds")

