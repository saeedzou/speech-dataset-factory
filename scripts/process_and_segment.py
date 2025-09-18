import os
import argparse
import torch
import datetime
from tqdm import tqdm
from pyannote.audio import Pipeline

# Local module imports (assuming they are in the correct path)
from src.models.diarization import speaker_diarization
from src.models.silero_vad import SileroVAD
from src.models.source_separation import source_separation, Predictor
from src.utils import write_mp3, save_speaker_embeddings
from src.preprocessing.standardization import standardization
from src.preprocessing.merge_vad import cut_by_speaker_label

def main(args):
    """
    Main function to process and segment audio files.
    """
    # --- 1. Setup Models ---
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Using device: {device_name}")

    # Source separation model config
    ss_cfg = {
        "model_path": "pretrained_models/UVR-MDX-NET-Inst_HQ_3.onnx",
        "denoise": True,
        "margin": 44100,
        "chunks": 15,
        "n_fft": 6144,
        "dim_t": 8,
        "dim_f": 3072
    }
    
    separate_predictor = Predictor(args=ss_cfg, device=device_name)
    
    # VAD and Diarization models
    vad = SileroVAD(device=device)
    dia_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    dia_pipeline.to(device)

    # --- 2. Prepare Directories ---
    os.makedirs(args.audio_dir, exist_ok=True)
    os.makedirs(args.embeddings_dir, exist_ok=True)

    # --- 3. Process Audio Files ---
    audio_files = [os.path.join(args.raw_dir, file) for file in os.listdir(args.raw_dir) if file.endswith('.mp3')]
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pbar = tqdm(audio_files, desc=f"Processing audio files", position=0)
    for audio_path in pbar:
        embedding_path = os.path.join(args.embeddings_dir, f"{os.path.splitext(os.path.basename(audio_path))[0]}.npy")
        if os.path.exists(embedding_path):
            continue
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pbar.set_postfix_str(f"[{current_time}] {os.path.basename(audio_path)}: Standardization...")
        # Standardize audio (load, resample, etc.)
        audio_original = standardization(audio_path)
        
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pbar.set_postfix_str(f"[{current_time}] Source Separation...")
        audio_cleaned = source_separation(separate_predictor, audio_original.copy())

        # Diarization is typically more robust on cleaner speech
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pbar.set_postfix_str(f"[{current_time}] Speaker Diarization...")
        speakerdia, embeddings = speaker_diarization(audio_cleaned, dia_pipeline, device)

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pbar.set_postfix_str(f"[{current_time}] VAD...")
        vad_list = vad.vad(speakerdia, audio_cleaned)


        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pbar.set_postfix_str(f"[{current_time}] Cut by speaker labels...")
        segment_list = cut_by_speaker_label(vad_list)

        # Determine which audio source to use for cutting final utterances
        source_for_cutting = audio_cleaned if args.cut_from_cleaned else audio_original
        
        for segment in tqdm(segment_list, desc="Saving segments", position=1, leave=False):
            index = segment['index']
            start_time = int(segment['start'] * source_for_cutting['sample_rate'])
            end_time = int(segment['end'] * source_for_cutting['sample_rate'])
            speaker_label = segment['speaker'].split('_')[-1]
            
            # Extract segment from the chosen source
            segment_audio = source_for_cutting['waveform'][start_time:end_time]
            
            # Construct output path
            base_filename = os.path.splitext(os.path.basename(audio_path))[0]
            output_path = os.path.join(args.audio_dir, f"{base_filename}_{speaker_label}_{index}.mp3")
            
            write_mp3(output_path, segment_audio, source_for_cutting['sample_rate'])

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pbar.set_postfix_str(f"[{current_time}] Saving embeddings...")
        save_speaker_embeddings(embeddings, os.path.splitext(os.path.basename(audio_path))[0], args.embeddings_dir)

    print("\nProcessing complete. ✨")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process long audio files into short, diarized utterances."
    )

    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Root directory containing the raw, long audio files (.mp3)."
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Output directory to save the segmented audio utterances."
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        required=True,
        help="Output directory to save the speaker embeddings."
    )
    parser.add_argument(
        "--cut_from_cleaned",
        action='store_true', # Creates a boolean flag
        help="If specified, cut final utterances from the cleaned (source-separated) audio instead of the original."
    )
    
    args = parser.parse_args()
    main(args)