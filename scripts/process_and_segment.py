import os
import argparse
import torch
import datetime
from tqdm import tqdm
from pyannote.audio import Pipeline
import json # Import json module

# Local module imports (assuming they are in the correct path)
from src.models.diarization import speaker_diarization
from src.models.silero_vad import SileroVAD
from src.models.source_separation import source_separation, Predictor
from src.utils import write_mp3, save_speaker_embeddings
from src.preprocessing.standardization import standardization
from src.preprocessing.merge_vad import cut_by_speaker_label
import gc

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def read_manifest(manifest_path):
    """Reads a JSONL manifest file line by line into a list of dicts."""
    data = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping malformed line: {e}")
    return data

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
    dia_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
    dia_pipeline.to(device)

    # --- 2. Prepare Directories ---
    os.makedirs(args.clips_24k_dir, exist_ok=True)
    os.makedirs(args.clips_16k_dir, exist_ok=True)
    os.makedirs(args.embeddings_dir, exist_ok=True)
    # os.makedirs(args.clean_dir, exist_ok=True)

    # Open manifest file for writing
    manifest_file = open(args.output_manifest, 'a')

    # --- 3. Process Audio Files ---
    # Read audio files from the input manifest
    input_data = read_manifest(args.input_manifest)
    audio_files = [entry['audio_filepath'] for entry in input_data]
    import random
    random.seed(42)  # set the seed
    random.shuffle(audio_files)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pbar = tqdm(audio_files, desc=f"Processing audio files", position=0)
    for audio_path in pbar:
        embedding_path = os.path.join(args.embeddings_dir, f"{os.path.splitext(os.path.basename(audio_path))[0]}.npy")
        if os.path.exists(embedding_path):
            continue
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pbar.set_postfix_str(f"[{current_time}] {os.path.basename(audio_path)}: Standardization...")
        # Standardize audio (load, resample, etc.)
        audio_processed = standardization(audio_path)
        audio_original_24k = {
            "waveform": audio_processed["waveform"],
            "name": audio_processed["name"],
            "sample_rate": audio_processed["sample_rate"]
        }
        audio_original_16k = {
            "waveform": audio_processed["raw_16k_waveform"],
            "name": audio_processed["name"],
            "sample_rate": audio_processed["raw_16k_sample_rate"]
        }
        
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pbar.set_postfix_str(f"[{current_time}] Source Separation...")
        audio_cleaned_24k = source_separation(separate_predictor, audio_original_24k.copy())
        clear_gpu()

        
        # # Save the cleaned 24kHz audio
        # cleaned_24k_output_path = os.path.join(args.clean_dir, f"{os.path.splitext(os.path.basename(audio_path))[0]}_cleaned_24k.mp3")
        # write_mp3(cleaned_24k_output_path, audio_cleaned_24k['waveform'], audio_cleaned_24k['sample_rate'])

        # # Save the original 16kHz mono audio (without source separation)
        # original_16k_output_path = os.path.join(args.clean_dir, f"{os.path.splitext(os.path.basename(audio_path))[0]}_original_16k.mp3")
        # write_mp3(original_16k_output_path, audio_original_16k['waveform'], audio_original_16k['sample_rate'])

        # Diarization is typically more robust on cleaner speech
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pbar.set_postfix_str(f"[{current_time}] Speaker Diarization...")
        speakerdia, embeddings = speaker_diarization(audio_cleaned_24k, dia_pipeline, device)
        clear_gpu()


        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pbar.set_postfix_str(f"[{current_time}] VAD...")
        vad_list = vad.vad(speakerdia, audio_cleaned_24k)


        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pbar.set_postfix_str(f"[{current_time}] Cut by speaker labels...")
        segment_list = cut_by_speaker_label(vad_list)

        for segment in tqdm(segment_list, desc="Saving segments", position=1, leave=False):
            index = segment['index']
            start_time_24k = int(segment['start'] * audio_cleaned_24k['sample_rate'])
            end_time_24k = int(segment['end'] * audio_cleaned_24k['sample_rate'])
            start_time_16k = int(segment['start'] * audio_original_16k['sample_rate'])
            end_time_16k = int(segment['end'] * audio_original_16k['sample_rate'])
            speaker_label = segment['speaker'].split('_')[-1]
            
            # Extract segment from the chosen 24kHz source
            segment_audio_24k = audio_cleaned_24k['waveform'][start_time_24k:end_time_24k]
            
            # Construct output path for 24kHz clip
            base_filename = os.path.splitext(os.path.basename(audio_path))[0]
            output_path_24k = os.path.join(args.clips_24k_dir, f"{base_filename}_{speaker_label}_{index}.mp3")
            
            write_mp3(output_path_24k, segment_audio_24k, audio_cleaned_24k['sample_rate'])

            # Extract segment from the 16kHz source
            segment_audio_16k = audio_original_16k['waveform'][start_time_16k:end_time_16k]

            # Construct output path for 16kHz clip
            output_path_16k = os.path.join(args.clips_16k_dir, f"{base_filename}_{speaker_label}_{index}.mp3")

            write_mp3(output_path_16k, segment_audio_16k, audio_original_16k['sample_rate'])

            # Write to manifest file
            manifest_entry = {
                "audio_filepath": output_path_24k, # This is the 24khz cleaned audio segment
                "audio_filepath_original_16k": output_path_16k, # This is the 16khz original audio segment
                "duration": segment['end'] - segment['start'],
                "start": segment['start'],
                "end": segment['end']
            }
            manifest_file.write(json.dumps(manifest_entry) + '\n')

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pbar.set_postfix_str(f"[{current_time}] Saving embeddings...")
        save_speaker_embeddings(embeddings, os.path.splitext(os.path.basename(audio_path))[0], args.embeddings_dir)
        del embeddings, speakerdia, audio_cleaned_24k, audio_original_24k, audio_original_16k
        clear_gpu()
    
    manifest_file.close() # Close the output manifest file

    print("\nProcessing complete. ✨")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process long audio files into short, diarized utterances."
    )

    parser.add_argument(
        "--input_manifest",
        type=str,
        required=True,
        help="Path to the input JSONL manifest file containing audio files to process."
    )
    parser.add_argument(
        "--clips_24k_dir",
        type=str,
        required=True,
        help="Output directory to save the segmented audio utterances at 24kHz."
    )
    parser.add_argument(
        "--clips_16k_dir",
        type=str,
        required=True,
        help="Output directory to save the segmented audio utterances at 16kHz."
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        required=True,
        help="Output directory to save the speaker embeddings."
    )

    # parser.add_argument(
    #     "--clean_dir",
    #     type=str,
    #     required=True,
    #     help="Output directory to save the source separated raw audio files."
    # )
    parser.add_argument(
        "--output_manifest",
        type=str,
        required=True,
        help="Path to the output JSONL manifest file."
    )
    
    args = parser.parse_args()
    main(args)
