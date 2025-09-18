import os
import numpy as np
from pydub import AudioSegment

def write_mp3(path, x, sr):
    """Convert numpy array to MP3."""
    try:
        # Ensure x is in the correct format and normalize if necessary
        if x.dtype != np.int16:
            # Normalize the array to fit in int16 range if it's not already int16
            x = np.int16(x / np.max(np.abs(x)) * 32767)

        # Create audio segment from numpy array
        audio = AudioSegment(
            x.tobytes(), frame_rate=sr, sample_width=x.dtype.itemsize, channels=1
        )
        # Export as MP3 file
        audio.export(path, format="mp3")
    except Exception as e:
        print(e)
        print("Error: Failed to write MP3 file.")

def save_speaker_embeddings(embeddings, filename, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{filename}.npy")
    np.save(path, embeddings)