import os
import json
import argparse
import torch
from tqdm import tqdm

def transcribe_with_nemo(args, model, manifest_data, output_file_handle):
    """Handles the transcription process using a NeMo ASR model."""
    audio_filepaths = [entry['audio_filepath'] for entry in manifest_data]
    total_files = len(audio_filepaths)
    
    # Use save_iterations as chunk_size, or process all at once if <= 0
    chunk_size = args.save_iterations if args.save_iterations > 0 else total_files
    
    with tqdm(total=total_files, desc="Transcribing (NeMo)") as pbar:
        for i in range(0, total_files, chunk_size):
            chunk_end = min(i + chunk_size, total_files)
            
            filepaths_chunk = audio_filepaths[i:chunk_end]
            manifest_chunk = manifest_data[i:chunk_end]

            if not filepaths_chunk:
                continue

            # Perform transcription on the current chunk
            transcriptions_chunk = model.transcribe(
                audio=filepaths_chunk,
                batch_size=args.batch_size,
                verbose=False
            )

            for entry, text in zip(manifest_chunk, transcriptions_chunk):
                entry['pred_text'] = text.text
                output_file_handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            output_file_handle.flush()
            pbar.update(len(filepaths_chunk))
            tqdm.write(f"PROGRESS: Saved manifest for {pbar.n}/{total_files} files.")

def transcribe_with_whisper(args, model, manifest_data, output_file_handle):
    """
    Handles the transcription process using a faster-whisper model,
    saving progress in chunks.
    """
    total_files = len(manifest_data)
    chunk_size = args.save_iterations if args.save_iterations > 0 else total_files

    with tqdm(total=total_files, desc="Transcribing (Whisper)") as pbar:
        for i in range(0, total_files, chunk_size):
            chunk_end = min(i + chunk_size, total_files)
            manifest_chunk = manifest_data[i:chunk_end]

            if not manifest_chunk:
                continue

            results_chunk = []
            # Process each file in the chunk
            for entry in manifest_chunk:
                audio_path = entry['audio_filepath']
                try:
                    segments, _ = model.transcribe(
                        audio_path, 
                        beam_size=args.beam_size, 
                        language=args.language, 
                        vad_filter=args.vad_filter
                    )
                    transcription = " ".join([s.text for s in segments]).strip()
                    entry['pred_text'] = transcription
                    results_chunk.append(entry)
                except Exception as e:
                    tqdm.write(f"ERROR: Could not process {os.path.basename(audio_path)}: {e}")
            
            # Write the results of the entire chunk to the file
            for result_entry in results_chunk:
                output_file_handle.write(json.dumps(result_entry, ensure_ascii=False) + '\n')

            output_file_handle.flush()
            pbar.update(len(manifest_chunk))
            tqdm.write(f"PROGRESS: Saved manifest for {pbar.n}/{total_files} files.")


def main(args):
    """
    Transcribes audio files from a manifest using either NeMo or faster-whisper,
    with resume capability and an option to specify a parent directory for audio files.
    """
    # 1. Setup device and load the appropriate ASR model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} 💻")

    model = None
    if args.model_type == 'nemo':
        from nemo.collections.asr.models import ASRModel
        print("Loading NeMo model...")
        try:
            model = ASRModel.restore_from(restore_path=args.model_path, map_location=torch.device(device))
        except Exception as e:
            print(f"Failed to load with restore_from: {e}. Trying from_pretrained for NGC models...")
            model = ASRModel.from_pretrained(model_name=args.model_path, map_location=torch.device(device))
        model.eval()
    elif args.model_type == 'whisper':
        from faster_whisper import WhisperModel
        print(f"Loading faster-whisper model with compute type '{args.compute_type}'...")
        model = WhisperModel(args.model_path, device=device, compute_type=args.compute_type)
    
    print(f"Model '{os.path.basename(args.model_path)}' loaded successfully. ✅")

    # 2. Resume Logic: Check for already processed files
    processed_filepaths = set()
    if os.path.exists(args.output_manifest) and not args.overwrite:
        print(f"Output manifest '{args.output_manifest}' found. Attempting to resume.")
        with open(args.output_manifest, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                try:
                    entry = json.loads(line)
                    if 'audio_filepath' in entry:
                        processed_filepaths.add(entry['audio_filepath'])
                except json.JSONDecodeError:
                    continue
        if processed_filepaths:
            print(f"Found {len(processed_filepaths)} already processed files to skip.")

    # 3. Read input manifest and construct full audio paths if needed
    full_manifest_data = []
    with open(args.input_manifest, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            full_manifest_data.append(json.loads(line))

    if args.audio_dir:
        print(f"Prepending parent directory '{args.audio_dir}' to all audio filepaths.")
        for entry in full_manifest_data:
            # Re-join even if absolute, to normalize path separators
            entry['audio_filepath'] = os.path.join(args.audio_dir, os.path.basename(entry['audio_filepath']))
    
    # 4. Filter the manifest to only include unprocessed files
    manifest_data_to_process = [
        entry for entry in full_manifest_data 
        if entry['audio_filepath'] not in processed_filepaths
    ]
    
    if not manifest_data_to_process:
        print("🎉 All files have already been processed. Nothing to do.")
        return

    print(f"Found {len(full_manifest_data)} total files, {len(manifest_data_to_process)} remaining to be transcribed.")

    # 5. Transcribe and write results
    file_mode = 'a' if os.path.exists(args.output_manifest) and not args.overwrite else 'w'
    with open(args.output_manifest, file_mode, encoding='utf-8') as f_out:
        if args.model_type == 'nemo':
            transcribe_with_nemo(args, model, manifest_data_to_process, f_out)
        elif args.model_type == 'whisper':
            transcribe_with_whisper(args, model, manifest_data_to_process, f_out)

    print(f"\n🎉 Transcription complete. Output saved to '{args.output_manifest}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Transcribe audio files from a manifest using NeMo or faster-whisper."
    )
    
    # --- General Arguments ---
    parser.add_argument("--model_type", type=str, required=True, choices=['nemo', 'whisper'], help="The type of model to use for transcription.")
    parser.add_argument("--model_path", type=str, required=True, help="Path or name of the ASR model. For NeMo: path to .nemo file or NGC model name. For Whisper: path to the faster-whisper model directory.")
    parser.add_argument("--input_manifest", type=str, required=True, help="Path to the input manifest file (JSONL format).")
    parser.add_argument("--output_manifest", type=str, required=True, help="Path to save the output manifest file.")
    parser.add_argument("--audio_dir", type=str, default=None, help="Optional parent directory for audio files. If provided, paths in manifest are treated as relative.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output manifest if it exists, instead of resuming.")
    parser.add_argument("--save_iterations", type=int, default=32, help="Save manifest every N files. If <=0, saves only at the end.")

    # --- NeMo Specific Arguments ---
    nemo_group = parser.add_argument_group('NeMo Specific Options')
    nemo_group.add_argument("--batch_size", type=int, default=32, help="Batch size for NeMo transcription. Adjust based on VRAM.")
    
    # --- Whisper Specific Arguments ---
    whisper_group = parser.add_argument_group('Whisper Specific Options')
    whisper_group.add_argument("--compute_type", type=str, default="float16", choices=['float16', 'int8_float16', 'int8', 'float32'], help="Compute type for faster-whisper model.")
    whisper_group.add_argument("--language", type=str, default=None, help="Language code for transcription (e.g., 'en', 'fa'). If None, Whisper will auto-detect.")
    whisper_group.add_argument("--beam_size", type=int, default=5, help="Beam size for Whisper decoding.")
    whisper_group.add_argument("--vad_filter", action="store_true", help="Enable VAD (Voice Activity Detection) filter in Whisper to remove silence.")
    
    args = parser.parse_args()
    main(args)
