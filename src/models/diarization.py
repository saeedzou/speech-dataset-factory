import torch
import pandas as pd

def speaker_diarization(audio, dia_pipeline, device):
    """
    Perform speaker diarization on the given audio.

    Args:
        audio (dict): A dictionary containing the audio waveform and sample rate.
        dia_pipeline (callable): The diarization pipeline to be used.
        device (str): The device to run the model on (e.g., 'cpu' or 'cuda').

    Returns:
        pd.DataFrame: A dataframe containing segments with speaker labels.
    """

    waveform = torch.tensor(audio["waveform"]).to(device)
    waveform = torch.unsqueeze(waveform, 0)

    segments, embeddings = dia_pipeline(
        {
            "waveform": waveform,
            "sample_rate": audio["sample_rate"],
            "channel": 0,
        },
        return_embeddings=True
    )

    diarize_df = pd.DataFrame(
        segments.itertracks(yield_label=True),
        columns=["segment", "label", "speaker"],
    )
    diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
    diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)

    return diarize_df, embeddings