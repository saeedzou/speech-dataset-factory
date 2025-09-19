import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperProcessor
from torchaudio.transforms import Resample
import os
from tqdm import tqdm
import json
import argparse

# ----------------------------- #
#         Load Components         #
# ----------------------------- #

def load_model_and_tokenizer(model_name="openai/whisper-large-v3"):
    """Loads the Whisper model, processor, and tokenizer."""
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name, attn_implementation="sdpa")
    tokenizer = WhisperTokenizer.from_pretrained(model_name)
    return processor, model, tokenizer

# ----------------------------- #
#      Load and Preprocess       #
# ----------------------------- #

def load_and_preprocess_audio(paths, target_sample_rate=16000):
    """Loads audio files, resamples them to the target rate, and returns a list of waveforms."""
    waveforms = []
    valid_paths = []
    for path in paths:
        try:
            waveform, sr = torchaudio.load(path)
            if sr != target_sample_rate:
                resampler = Resample(orig_freq=sr, new_freq=target_sample_rate)
                waveform = resampler(waveform)
            waveforms.append(waveform[0])  # mono
            valid_paths.append(path)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
    if not waveforms:
        return None, []
    return pad_waveforms(waveforms), valid_paths

def pad_waveforms(waveforms):
    """Pads a list of waveforms to the same length and stacks them into a tensor."""
    max_len = max(w.shape[0] for w in waveforms)
    padded_waveforms = [
        torch.nn.functional.pad(w, (0, max_len - w.shape[0])) for w in waveforms
    ]
    return torch.stack(padded_waveforms)

# ----------------------------- #
#      Language Detection       #
# ----------------------------- #

def detect_language(model, tokenizer, input_features, possible_languages=None):
    """Detects the language of the input audio features."""
    language_tokens = [t for t in tokenizer.additional_special_tokens if len(t) == 6]
    if possible_languages is not None:
        language_tokens = [t for t in language_tokens if t[2:-2] in possible_languages]
        if len(language_tokens) < len(possible_languages):
            raise RuntimeError(f'Some languages in {possible_languages} did not have associated language tokens')

    language_token_ids = tokenizer.convert_tokens_to_ids(language_tokens)
    decoder_input_ids = torch.tensor([[50258]] * input_features.shape[0]).to(input_features.device)
    logits = model(input_features, decoder_input_ids=decoder_input_ids).logits

    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[language_token_ids] = False
    logits[:, :, mask] = -float('inf')

    output_probs = logits.softmax(dim=-1).cpu()

    return [
        {
            lang: output_probs[i, 0, token_id].item()
            for token_id, lang in zip(language_token_ids, language_tokens)
        }
        for i in range(logits.shape[0])
    ]

# ----------------------------- #
#        Helper Functions       #
# ----------------------------- #

def append_to_manifest(filepath, records):
    """Appends a list of records to a JSON lines file."""
    with open(filepath, 'a', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

# ----------------------------- #
#            Main               #
# ----------------------------- #

def main(input_manifest_path, output_manifest_path, batch_size, save_iterations):
    """Main function to run the language detection pipeline."""
    # Load manifest data from JSON lines file
    manifest_data = []
    print(f"Reading manifest from: {input_manifest_path}")
    try:
        with open(input_manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if 'audio_filepath' not in record:
                        print(f"Skipping line, missing 'audio_filepath': {line.strip()}")
                        continue
                    manifest_data.append(record)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: Input manifest file not found at {input_manifest_path}")
        return

    if not manifest_data:
        print("No valid audio filepaths found in the manifest.")
        return

    # Create a lookup map for original records and a list of paths to process
    record_map = {rec['audio_filepath']: rec for rec in manifest_data}
    audio_paths = [rec['audio_filepath'] for rec in manifest_data]
    total = len(audio_paths)

    # Load model components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    processor, model, tokenizer = load_model_and_tokenizer()
    model = model.to(device)

    # Clear the output file at the start
    with open(output_manifest_path, 'w', encoding='utf-8'):
        pass

    records_to_write = []
    print("Processing audio files in batches...")
    pbar = tqdm(range(0, total, batch_size), desc="Batches")
    for batch_idx, i in enumerate(pbar):
        batch_paths = audio_paths[i:i + batch_size]
        batch_waveform, valid_paths = load_and_preprocess_audio(batch_paths)

        if batch_waveform is None:
            continue

        inputs = processor(batch_waveform.numpy(), sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        language_probs = detect_language(model, tokenizer, input_features, {'en', 'fa'})

        for path, probs in zip(valid_paths, language_probs):
            fa_prob = probs.get("<|fa|>", 0.0)
            en_prob = probs.get("<|en|>", 0.0)
            detected_lang = "fa" if fa_prob >= en_prob else "en"
            
            # Find original record, update it, and add to write buffer
            original_record = record_map.get(path)
            if original_record:
                updated_record = original_record.copy()
                updated_record.update({
                    "lang": detected_lang,
                    "fa_lang_prob": fa_prob,
                    "en_lang_prob": en_prob
                })
                records_to_write.append(updated_record)

        # Check if it's time to save progress
        if save_iterations > 0 and (batch_idx + 1) % save_iterations == 0:
            pbar.set_postfix_str(f"Saving progress... ({len(records_to_write)} records)")
            append_to_manifest(output_manifest_path, records_to_write)
            records_to_write = [] # Reset buffer

    # Final save for any remaining records
    if records_to_write:
        append_to_manifest(output_manifest_path, records_to_write)

    print(f"Language detection completed. Results saved to {output_manifest_path}")

# ----------------------------- #
#        Run the Script         #
# ----------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform language detection on audio files specified in a JSON manifest.")
    parser.add_argument('--input_manifest', type=str, required=True, help='Path to the input manifest file (JSON lines).')
    parser.add_argument('--output_manifest', type=str, required=True, help='Path to save the output manifest file with language detection results.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing.')
    parser.add_argument('--save_iterations', type=int, default=64, help='Save progress every N batches. If 0, saves only at the end. (default: 0)')

    args = parser.parse_args()

    main(
        input_manifest_path=args.input_manifest,
        output_manifest_path=args.output_manifest,
        batch_size=args.batch_size,
        save_iterations=args.save_iterations
    )