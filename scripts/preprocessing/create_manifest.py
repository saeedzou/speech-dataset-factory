import os
import json
import argparse
import multiprocessing
from pydub.utils import mediainfo
from tqdm import tqdm

def process_file(filepath):
    """
    Processes a single audio file to get its duration.

    Args:
        filepath (str): The full path to the audio file.

    Returns:
        tuple(str, float) or None: A tuple of (filepath, duration), or None on error.
    """
    try:
        # Extract media information using ffprobe
        info = mediainfo(filepath)
        duration = float(info['duration'])
        # Return the original filepath along with its duration
        return (filepath, duration)
    except Exception:
        # Suppress printing errors for cleaner tqdm output
        return None

def create_manifest(audio_dir, embedding_dir, manifest_path, num_workers):
    """
    Scans a directory for .mp3 files, extracts their duration in parallel,
    and writes metadata with relative paths to a JSON manifest file.

    Args:
        audio_dir (str): The path to the directory containing audio files.
        manifest_path (str): The path to the output manifest file.
        num_workers (int): The number of parallel processes to use.
    """
    if not os.path.isdir(audio_dir):
        print(f"Error: Directory not found at {audio_dir}")
        return

    print(f"Scanning directory: {audio_dir}")
    print(f"Writing manifest to: {manifest_path}")

    # 1. Collect all full file paths first
    full_filepaths = []
    for root, _, filenames in os.walk(audio_dir):
        for filename in filenames:
            if filename.lower().endswith('.mp3'):
                full_filepaths.append(os.path.join(root, filename))
    
    print(f"Found {len(full_filepaths)} .mp3 files to process.")

    # 2. Use a multiprocessing Pool to process files in parallel
    results = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        with tqdm(total=len(full_filepaths), desc="Processing files") as pbar:
            # imap_unordered is efficient for showing progress as tasks complete
            for result in pool.imap_unordered(process_file, full_filepaths):
                if result:
                    results.append(result)
                pbar.update()
    results.sort(key=lambda x: os.path.basename(x[0]))

    # 3. Write results to the manifest file using paths relative to the current directory
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for file_path, duration in results:
            # The path from os.walk is already relative to the current working directory.
            # We normalize path separators to forward slashes for consistency.
            entry = {
                'audio_filepath': file_path.replace(os.sep, '/'),
                'duration': duration,
                'speaker_id': os.path.basename(file_path).split('_')[1],
                'embedding_file': os.path.join(embedding_dir, 
                                               "_".join(os.path.basename(file_path.replace('.mp3', '')).split('_')[:-2]) + ".npy")
            }
            f.write(json.dumps(entry) + '\n')
    
    print("Manifest creation complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create a JSON manifest file with relative paths for a directory of MP3s."
    )
    
    parser.add_argument(
        "--audio_dir",
        required=True,
        type=str,
        help="Path to the directory containing .mp3 files. Output paths will be relative to this."
    )

    parser.add_argument(
        "--embedding_dir",
        required=True,
        type=str,
        help="Path to the directory containing embedding files."
    )

    parser.add_argument(
        "--manifest_path",
        required=True,
        type=str,
        help="Path to the output JSON manifest file."
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes. Defaults to the number of CPU cores."
    )
    
    args = parser.parse_args()
    create_manifest(args.audio_dir, args.embedding_dir, args.manifest_path, args.num_workers)

