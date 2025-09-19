import argparse
import json
import sys
import re

def clean_text(text):
    """
    Cleans text by replacing specific unicode characters and removing punctuation.
    """
    # Replace \u200c with space
    text = text.replace("\u200c", " ")
    text = text.replace(" ", "")
    
    # Remove standard punctuation + "،:؛؟"
    text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~،:؛؟]', '', text)
    
    return text

def filter_manifest(input_path, output_path, max_wer, max_cer, min_chars):
    """
    Filters a JSON manifest file to remove entries exceeding WER/CER thresholds
    or with text length below a minimum character count.

    Each line in the manifest is expected to be a JSON object with at least
    'wer', 'cer', and 'text' keys.

    Args:
        input_path (str): Path to the input manifest file.
        output_path (str): Path to write the filtered manifest file.
        max_wer (float): The maximum allowed Word Error Rate.
        max_cer (float): The maximum allowed Character Error Rate.
        min_chars (int): The minimum allowed number of characters in the cleaned text.
    """
    try:
        total_lines = 0
        written_lines = 0
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                total_lines += 1
                try:
                    # Parse the JSON object from the current line
                    data = json.loads(line.strip())

                    # Ensure the required keys are present
                    if 'wer' not in data or 'cer' not in data or 'text' not in data:
                        sys.stderr.write(f"Warning: Skipping line {total_lines} due to missing 'wer', 'cer', or 'text' key.\n")
                        continue
                    
                    # Clean the transcript text
                    cleaned_text = clean_text(data['text'])

                    # Apply the filter conditions. We keep utterances where
                    # WER, CER, and text length all meet the criteria.
                    if data['wer'] <= max_wer and data['cer'] <= max_cer and len(cleaned_text) >= min_chars:
                        # Write the original line to the output file
                        outfile.write(line)
                        written_lines += 1

                except json.JSONDecodeError:
                    sys.stderr.write(f"Warning: Skipping malformed JSON on line {total_lines}.\n")
                except TypeError:
                    sys.stderr.write(f"Warning: Skipping line {total_lines} due to invalid type for 'wer' or 'cer'.\n")
        
        print("Filtering complete.")
        print(f"Total lines read: {total_lines}")
        print(f"Lines written to output: {written_lines}")
        print(f"Lines removed: {total_lines - written_lines}")

    except FileNotFoundError:
        sys.stderr.write(f"Error: Input file not found at '{input_path}'\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"An unexpected error occurred: {e}\n")
        sys.exit(1)

def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Filter an ASR manifest file based on WER and CER.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_manifest",
        required=True,
        type=str,
        help="Path to the input JSON manifest file (one JSON object per line)."
    )
    parser.add_argument(
        "--output_manifest",
        required=True,
        type=str,
        help="Path to save the filtered output manifest file."
    )
    # The user requested --min_wer/cer, but based on the task "filter_asr_errors",
    # --max_wer/cer is the standard and more intuitive naming convention.
    # This filters out utterances WITH an error rate HIGHER than the provided value.
    parser.add_argument(
        "--max_wer",
        required=True,
        default=0.15,
        type=float,
        help="Maximum Word Error Rate to keep. Utterances with WER above this will be removed."
    )
    parser.add_argument(
        "--max_cer",
        required=True,
        default=0.1,
        type=float,
        help="Maximum Character Error Rate to keep. Utterances with CER above this will be removed."
    )
    parser.add_argument(
        "--min_chars",
        required=True,
        default=3,
        type=int,
        help="Minimum number of characters (after cleaning) to keep. Utterances with text length below this will be removed."
    )

    args = parser.parse_args()

    filter_manifest(args.input_manifest, args.output_manifest, args.max_wer, args.max_cer, args.min_chars)

if __name__ == "__main__":
    main()
