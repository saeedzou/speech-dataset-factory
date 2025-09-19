import re
import json
import argparse
from jiwer import wer, cer

def clean_text(text):
    # Replace \u200c with space
    text = text.replace("\u200c", " ")
    
    # Remove standard punctuation + "،:؛؟"
    text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~،:؛؟]', '', text)
    
    return text

def main():
    parser = argparse.ArgumentParser(description="Compare two JSONL manifests and compute WER/CER per utterance.")
    parser.add_argument("manifest1", type=str, help="Path to first manifest file (predictions).")
    parser.add_argument("manifest2", type=str, help="Path to second manifest file (ground truth).")
    parser.add_argument("--output_manifest", type=str, required=True, help="Path to write output manifest with WER/CER.")
    args = parser.parse_args()

    # Load manifests into dicts keyed by audio_filepath
    def load_manifest(path):
        data = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                if item.get("pred_text"):  # skip empty pred_text
                    data[item["audio_filepath"]] = item
        return data

    manifest1_data = load_manifest(args.manifest1)
    manifest2_data = load_manifest(args.manifest2)

    # Compute WER and CER for matching audio files
    output_data = []
    for audio_path, item1 in manifest1_data.items():
        if audio_path in manifest2_data:
            item2 = manifest2_data[audio_path]

            pred_text = clean_text(item1["pred_text"])
            ref_text = clean_text(item2["pred_text"])

            if pred_text.strip() == "" or ref_text.strip() == "":
                continue  # skip if cleaned text is empty

            item1["wer"] = wer(ref_text, pred_text)
            item1["cer"] = cer(ref_text, pred_text)
            item1["text"] = ref_text
            item1["pred_text"] = pred_text
            output_data.append(item1)

    # Write output manifest
    with open(args.output_manifest, "w", encoding="utf-8") as out_f:
        for item in output_data:
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Processed {len(output_data)} utterances. Output saved to {args.output_manifest}")

if __name__ == "__main__":
    main()
