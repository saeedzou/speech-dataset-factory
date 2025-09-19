import argparse
import json
from tqdm import tqdm

def filter_manifest_by_language_prob(input_manifest, output_manifest, lang, min_prob):
    """
    Filters a JSON manifest file based on a minimum probability for a specified language.

    Args:
        input_manifest (str): Path to the input JSON lines manifest file.
        output_manifest (str): Path to write the filtered JSON lines manifest file.
        lang (str): The language code to check (e.g., 'fa', 'en').
        min_prob (float): The minimum probability threshold. Samples below this will be removed.
    """
    prob_key = f"{lang}_lang_prob"
    records_read = 0
    records_written = 0

    try:
        # First, count the total number of lines for the progress bar
        with open(input_manifest, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f)

        print(f"Filtering manifest '{input_manifest}'...")
        print(f"Keeping samples where language is '{lang}' with probability >= {min_prob}")

        with open(input_manifest, 'r', encoding='utf-8') as f_in, \
             open(output_manifest, 'w', encoding='utf-8') as f_out:

            for line in tqdm(f_in, total=total_lines, desc="Processing samples"):
                records_read += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"\nWarning: Skipping malformed JSON line #{records_read}: {line.strip()}")
                    continue

                # Check if the probability key exists and meets the threshold
                lang_prob = record.get(prob_key)
                if lang_prob is not None and lang_prob >= min_prob:
                    f_out.write(line)
                    records_written += 1

        print("\nFiltering complete.")
        print(f"Total records read: {records_read}")
        print(f"Records written to '{output_manifest}': {records_written}")
        print(f"Records removed: {records_read - records_written}")

    except FileNotFoundError:
        print(f"Error: Input manifest file not found at '{input_manifest}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter a manifest file by removing samples where a specific language's probability is below a threshold.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--input_manifest',
        type=str,
        required=True,
        help='Path to the input manifest file (JSON lines format).'
    )
    parser.add_argument(
        '--output_manifest',
        type=str,
        required=True,
        help='Path to save the filtered output manifest file.'
    )
    parser.add_argument(
        '--lang',
        type=str,
        required=True,
        help="The language code to filter by (e.g., 'fa', 'en')."
    )
    parser.add_argument(
        '--min_prob',
        type=float,
        required=True,
        help="The minimum probability for the specified language. Samples below this will be removed."
    )

    args = parser.parse_args()

    filter_manifest_by_language_prob(
        input_manifest=args.input_manifest,
        output_manifest=args.output_manifest,
        lang=args.lang,
        min_prob=args.min_prob
    )
