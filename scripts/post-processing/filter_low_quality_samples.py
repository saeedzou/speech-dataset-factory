import argparse
import json

def filter_manifest(input_manifest: str, output_manifest: str, min_snr: float, min_c50: float) -> None:
    """
    Filters entries in a JSON manifest based on minimum SNR and C50 thresholds.

    Args:
        input_manifest (str): Path to the input JSON manifest.
        output_manifest (str): Path to write the filtered JSON manifest.
        min_snr (float): Minimum acceptable SNR (default: 25).
        min_c50 (float): Minimum acceptable C50 (default: 30).
    """
    filtered_entries = []

    print(f"Reading manifest from: {input_manifest}")
    try:
        with open(input_manifest, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                snr = entry.get('average_snr', None)
                c50 = entry.get('average_c50', None)

                if snr is None or c50 is None:
                    print(f"Skipping entry missing SNR or C50: {entry.get('audio_filepath', 'unknown')}")
                    continue

                if snr >= min_snr and c50 >= min_c50:
                    filtered_entries.append(entry)
    except FileNotFoundError:
        print(f"Error: Manifest file not found at {input_manifest}")
        return

    print(f"Filtered {len(filtered_entries)} entries from {input_manifest}")

    print(f"Writing filtered manifest to: {output_manifest}")
    with open(output_manifest, 'w', encoding='utf-8') as f:
        for entry in filtered_entries:
            f.write(json.dumps(entry) + '\n')

    print("✅ Filtering complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter a JSON manifest to remove samples below specified SNR and C50 thresholds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input_manifest', type=str, required=True, help="Path to input JSON manifest")
    parser.add_argument('--output_manifest', type=str, required=True, help="Path to write filtered manifest")
    parser.add_argument('--min_snr', type=float, default=25.0, help="Minimum SNR threshold")
    parser.add_argument('--min_c50', type=float, default=30.0, help="Minimum C50 threshold")

    args = parser.parse_args()

    filter_manifest(args.input_manifest, args.output_manifest, args.min_snr, args.min_c50)
