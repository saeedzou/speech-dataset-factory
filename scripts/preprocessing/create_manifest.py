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
        filepath (str): The path to the audio file.

    Returns:
        dict or None: A dictionary with audio_filepath and duration, or None on error.
    """
    try:
        # Extract media information using ffprobe
        info = mediainfo(filepath)
        duration = float(info['duration'])

        # Create a JSON object for the audio file
        manifest_entry = {
            'audio_filepath': filepath,
            'duration': duration
        }
        return manifest_entry
    except Exception as e:
        # Suppress printing errors for cleaner tqdm output, but you can re-enable if needed
        # print(f"Error processing file {filepath}: {e}")
        return None

def create_manifest(audio_dir, manifest_path, num_workers):
    """
    Scans a directory for .mp3 files, extracts their duration using pydub's
    mediainfo in parallel, and writes the metadata to a JSON manifest file.

    Each line in the manifest file is a separate JSON object.

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
    print(f"Using {num_workers} worker processes.")

    # 1. Collect all file paths first
    filepaths = []
    for root, _, filenames in os.walk(audio_dir):
        for filename in filenames:
            if filename.lower().endswith('.mp3'):
                filepaths.append(os.path.abspath(os.path.join(root, filename)))
    
    print(f"Found {len(filepaths)} .mp3 files to process.")

    results = []
    # 2. Use a multiprocessing Pool to process files in parallel
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use imap_unordered for progress bar compatibility and efficiency.
        # Wrap the iterator with tqdm to show progress.
        with tqdm(total=len(filepaths)) as pbar:
            for result in pool.imap_unordered(process_file, filepaths):
                if result:
                    results.append(result)
                pbar.update()

    # 3. Write results to the manifest file
    with open(manifest_path, 'w') as f:
        for entry in results:
            f.write(json.dumps(entry) + '\n')
    
    print("Manifest creation complete.")

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Create a JSON manifest file for a directory of MP3 audio files."
    )
    
    parser.add_argument(
        "--audio_dir",
        required=True,
        type=str,
        help="Path to the directory containing .mp3 files."
    )
    
    parser.add_argument(
        "--manifest_path",
        required=True,
        type=str,
        help="Path to the output JSON manifest file."
    )

    parser.add_argument(
        "--num_workers",
        required=False,
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes for parallel processing. Defaults to the number of CPU cores."
    )
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Call the main function with the parsed arguments
    create_manifest(args.audio_dir, args.manifest_path, args.num_workers)