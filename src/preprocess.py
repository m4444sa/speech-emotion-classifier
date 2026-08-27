import librosa
import numpy as np


TARGET_SR = 16000 #sample rate
TARGET_DURATION = 4.0  # seconds 

def load_and_process(filepath, target_sr=TARGET_SR, target_duration=TARGET_DURATION):
    y, sr = librosa.load(filepath, sr=target_sr)  
    target_len = int(target_sr * target_duration)

    if len(y) > target_len:
        y = y[:target_len]              # trim
    else:
        y = np.pad(y, (0, target_len - len(y)))  # pad with zeros

    return y