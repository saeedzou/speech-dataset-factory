import argparse
import json
import os
import numpy as np
from tqdm.auto import tqdm
from pyannote.audio import Model, Inference

# -------------------------------------------------- #
#                  Core Function                     #
# -------------------------------------------------- #

def calculate_audio_metrics_averages(audio_filepath: str, inference: Inference) -> tuple[float, float]:
    """
    Calculates the average Signal-to-Noise Ratio (SNR) and Clarity (C50) for a given audio file.
    
    Args:
        audio_filepath: Path to the audio file.
        inference: The pyannote.audio Inference instance.

    Returns:
        A tuple containing the average SNR and average C50. Returns (nan, nan) on failure.
    """
    if not os.path.exists(audio_filepath):
        print(f"Warning: File not found, skipping: {audio_filepath}")
        return (float('nan'), float('nan'))

    try:
        inference_output = inference(audio_filepath)
        snr_frames = [snr for _, (_, snr, _) in inference_output if snr is not None]
        c50_frames = [c50 for _, (_, _, c50) in inference_output if c50 is not None]

        average_snr = np.mean(snr_frames) if snr_frames else float('nan')
        average_c50 = np.mean(c50_frames) if c50_frames else float('nan')

        return (average_snr, average_c50)
    except Exception as e:
        print(f"Error processing {audio_filepath}: {e}")
        return (float('nan'), float('nan'))

# -------------------------------------------------- #
#                       Main                         #
# -------------------------------------------------- #

def main():
    """
    Processes an audio manifest to calculate and append SNR and C50 metrics.
    Supports resuming from an existing output manifest.
    """
    parser = argparse.ArgumentParser(
        description="Calculate SNR and C50 for audio files from a JSON-lines manifest."
    )
    parser.add_argument(
        "--input_manifest",
        required=True,
        type=str,
        help="Path to the input manifest file (JSON-lines format, each line has 'audio_filepath')."
    )
    parser.add_argument(
        "--output_manifest",
        required=True,
        type=str,
        help="Path for the output manifest file (JSON-lines format)."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output manifest if it exists, disabling the resume feature."
    )
    args = parser.parse_args()

    # --- Resume Logic ---
    processed_files = set()
    if not args.overwrite and os.path.exists(args.output_manifest):
        print(f"✅ Output manifest found. Attempting to resume from {args.output_manifest}...")
        try:
            with open(args.output_manifest, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    if 'audio_filepath' in entry:
                        processed_files.add(entry['audio_filepath'])
            print(f"Found {len(processed_files)} already processed files.")
        except (IOError, json.JSONDecodeError) as e:
            print(f"⚠️ Warning: Could not read existing output manifest for resuming. Starting fresh. Error: {e}")
            args.overwrite = True

    # --- Load Input Manifest & Filter ---
    tasks = []
    try:
        with open(args.input_manifest, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if 'audio_filepath' not in entry:
                        print(f"Skipping line without 'audio_filepath': {line.strip()}")
                        continue
                    if entry['audio_filepath'] not in processed_files:
                        tasks.append(entry)
                except json.JSONDecodeError:
                    print(f"Skipping malformed JSON line: {line.strip()}")
    except FileNotFoundError:
        print(f"❌ Error: Input manifest not found at {args.input_manifest}")
        return

    if not tasks:
        print("✅ No new files to process. Exiting.")
        return
        
    # --- Load Model ---
    print("Loading pyannote/brouhaha model...")
    model = Model.from_pretrained("pyannote/brouhaha")
    inference = Inference(model)

    # --- Processing ---
    file_mode = 'w' if args.overwrite else 'a'
    print(f"🚀 Processing {len(tasks)} new audio files...")
    
    with open(args.output_manifest, file_mode, encoding='utf-8') as f_out:
        for entry in tqdm(tasks, desc="Calculating audio metrics"):
            audio_filepath = entry['audio_filepath']
            avg_snr, avg_c50 = calculate_audio_metrics_averages(audio_filepath, inference)
            # Convert numpy floats to Python native floats for JSON serialization
            # and handle potential NaNs for clean JSON output
            if np.isnan(avg_snr):
                entry['average_snr'] = None
            else:
                entry['average_snr'] = float(avg_snr)  # Convert numpy float to Python float
                
            if np.isnan(avg_c50):
                entry['average_c50'] = None
            else:
                entry['average_c50'] = float(avg_c50)  # Convert numpy float to Python float
            
            f_out.write(json.dumps(entry) + '\n')
            
    print(f"🎉 Processing complete. Results saved to {args.output_manifest}")

# -------------------------------------------------- #
#                  Run the Script                    #
# -------------------------------------------------- #

if __name__ == "__main__":
    main()