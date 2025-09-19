# speech-dataset-factory

A configurable pipeline for creating speech datasets for tasks like Automatic Speech Recognition (ASR), Text-to-Speech (TTS), and Voice Cloning (VC). This project provides a series of processing steps to take raw audio files and turn them into a structured dataset of segmented audio with speaker labels.

## Features

- **Audio Standardization**: Normalizes audio to a consistent format (24kHz, 16-bit, mono).
- **Source Separation**: Isolates vocals from background music and noise.
- **Speaker Diarization**: Identifies and labels different speakers in the audio so that each utterance has only one speaker.
- **Voice Activity Detection (VAD)**: Further segments the diarized audio to more fine-grained utterances.
- **Configurable Segmentation**: Allows for customization of segment length and merging gaps.

## Project Structure

```bash
speech-dataset-factory/
├── scripts/
│   ├── preprocessing/
│   │   ├── calculate_asr_metrics.py
│   │   ├── create_manifest.py
│   │   ├── filter_language.py
│   │   └── filter_low_quality_samples.py
│   ├── asr.py
│   ├── cluster_speakers.py
│   ├── detect_language.py
│   └── process_and_segment.py
├── src/
│   ├── models/
│   │   ├── diarization.py
│   │   ├── silero_vad.py
│   │   └── source_separation.py
│   ├── preprocessing/
│   │   ├── merge_vad.py
│   │   └── standardization.py
│   └── utils.py
├── pretrained_models/
│   └── UVR-MDX-NET-Inst_HQ_3.onnx
├── data/
│   ├── raw/
│   │   └── (Your raw audio files)
│   ├── audio/
│   │   └── (Segmented audio output)
│   └── embeddings/
│       └── (Speaker embeddings output)
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/saeedzou/speech-dataset-factory.git
cd speech-dataset-factory
```

2. Install dependencies:

```bash
pip install -r requirements.txt
pip uninstall onnxruntime onnxruntime-gpu # To avoid conflicts with faster whisper
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

3. Download the pretrained model:

Download pretrained UVR model and place it in `pretrained_models`

```bash
wget https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_3.onnx -O pretrained_models/UVR-MDX-NET-Inst_HQ_3.onnx
```

## Usage

### Process and Segment Audio

Place your raw audio files (in `.mp3` format) into the `data/raw` directory. Then, run the main processing script:

```bash
python -m scripts.process_and_segment \
    --raw_dir data/raw \ # Directory containing raw audio files
    --audio_dir data/audio \ # Directory for segmented audio output
    --embeddings_dir data/embeddings # Directory for speaker embeddings output
    --cut_from_cleaned # If specified, cut segments from cleaned audio
```

### Create Manifest

To create a manifest file for your audio dataset, run the following command:

```bash
python -m scripts.create_manifest \
    --audio_dir data/audio \ # Directory containing segmented audio files
    --embeddings_dir data/embeddings \ # Directory containing speaker embeddings
    --output_manifest data/manifest.json # Path for the output manifest file
```

### Calculate Audio Metrics

To calculate audio metrics (SNR and C50) for your audio dataset, run the following command:

```bash
python -m scripts.calculate_quality_metrics \
    --input_manifest data/manifest.json \ # Path to the input manifest file
    --output_manifest data/manifest_metrics.json \ # Path for the output manifest file
    --overwrite # Overwrite the output manifest if it exists
```

Then remove low quality utterances based on their SNR and C50 metrics.

```bash
python -m scripts.preprocessing.filter_low_quality_samples \
    --input_manifest data/manifest_metrics.json \ # Path to the input manifest file with metrics
    --output_manifest data/manifest_metrics_filtered.json \ # Path for the output manifest file with filtered samples
    --min_snr 25 \ # Minimum SNR threshold (dB)
    --min_c50 30 # Minimum C50 threshold (dB)
```

### Detect Language

To detect the language of the audio segments, run the following command:

```bash
python -m scripts.detect_language \
    --input_manifest data/manifest_metrics_filtered.json \ # Path to the input manifest file with filtered samples
    --output_manifest data/manifest_metrics_filtered_language.json # Path for the output manifest file with detected language
```

Then, you can filter the manifest by language using the following command:

```bash
python -m scripts.preprocessing.filter_language \
    --input_manifest data/manifest_metrics_filtered_language.json \ # Path to the input manifest file with detected languages
    --output_manifest data/manifest_metrics_filtered_language_filtered.json \ # Path for the output manifest file with filtered samples
    --lang fa \ # Language code to filter by
    --min_prob 0.99 # Minimum probability for the specified language
```

### Get Unique Speakers across Files Using Clustering

Cluster speaker embeddings from multiple files using DBSCAN and assign consistent unique speaker IDs in an updated manifest. Run the following command:

```bash
python -m scripts.cluster_speakers \
    --input_manifest data/manifest_metrics_filtered_language_filtered.json \ # Path to the input manifest file with metrics
    --output_manifest data/manifest_metrics_filtered_language_filtered_speakers.json # Path for the output manifest file with clustered speaker IDs
```

### Transcribe Audio Segments

There are two model types available: `nemo` and `whisper`.

To transcribe audio segments using the `nemo` model, run the following command:

```bash
python -m scripts.asr \
    --model_type nemo \ # Specify the model type
    --model_path <path_to_nemo_model> \ # Path to the local Nemo model or huggingface model
    --input_manifest data/manifest_metrics_filtered_language_filtered_speakers.json \ # Path to the input manifest file with clustered speaker IDs
    --output_manifest data/manifest_metrics_filtered_language_filtered_speakers_transcribed_nemo.json \ # Path for the output manifest file with transcriptions
    --batch_size 32 \ # Specify the batch size
    --save_iterations 64 # Save manifest every N files. If <=0, saves only at the end.
```

To transcribe audio segments using the `whisper` model, run the following command (faster Whisper does not support batch size):

```bash
python -m scripts.asr \
    --model_type whisper \ # Specify the model type
    --model_path <path_to_whisper_model> \ # Path to the local Faster Whisper model or huggingface model
    --input_manifest data/manifest_metrics_filtered_language_filtered_speakers.json \ # Path to the input manifest file with clustered speaker IDs
    --output_manifest data/manifest_metrics_filtered_language_filtered_speakers_transcribed_whisper.json \ # Path for the output manifest file with transcriptions
    --save_iterations 64 # Save manifest every N files. If <=0, saves only at the end.
```

### Calculate ASR Metrics

After transcribing the audio segments with two different models, you can calculate the ASR metrics (Word Error Rate and Character Error Rate). Use the stronger model's predictions as a proxy for the weaker model's ground truth.

```bash
python -m scripts.preprocessing.calculate_asr_metrics \
    data/manifest_metrics_filtered_language_filtered_speakers_transcribed_whisper.json \
    data/manifest_metrics_filtered_language_filtered_speakers_transcribed_nemo.json \
    --output_manifest data/manifest_metrics_filtered_language_filtered_speakers_transcribed_metrics.json