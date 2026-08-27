import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.preprocess import load_and_process

DATA_DIR = "data" 


#RAVDESS filenames look like 03-01-06-01-02-01-12.wav
# the 3rd one maps to emotion
emotion_map = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

records = []

for actor_folder in os.listdir(DATA_DIR):
    actor_path = os.path.join(DATA_DIR, actor_folder)
    if not os.path.isdir(actor_path):
        continue
    for filename in os.listdir(actor_path):
        if not filename.endswith(".wav"):
            continue
        parts = filename.replace(".wav", "").split("-")
        emotion_code = parts[2]
        records.append({
            "filepath": os.path.join(actor_path, filename),
            "emotion_code": emotion_code,
            "emotion": emotion_map[emotion_code]
        })

df = pd.DataFrame(records)

print(df.shape)
print(df["emotion"].value_counts())

os.makedirs("data", exist_ok=True)
df.to_csv("data/labels.csv", index=False)

# building a pytorch dataset

EMOTION_LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
label_to_idx = {label: i for i, label in enumerate(EMOTION_LABELS)}

class RAVDESSDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio = load_and_process(row["filepath"])
        label = label_to_idx[row["emotion"]]
        return torch.tensor(audio, dtype=torch.float32), torch.tensor(label, dtype=torch.long)



class RAVDESSWav2Vec2Dataset(Dataset):
    def __init__(self, dataframe, feature_extractor):
        self.df = dataframe.reset_index(drop=True)
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio = load_and_process(row["filepath"])
        label = label_to_idx[row["emotion"]]

        inputs = self.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )

        return {
            "input_values": inputs["input_values"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }