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
```

3. Download the pretrained model:

Download pretrained UVR model and place it in `pretrained_models`

```bash
wget https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_3.onnx -O pretrained_models/UVR-MDX-NET-Inst_HQ_3.onnx
```

## Usage

Place your raw audio files (in `.mp3` format) into the `data/raw` directory. Then, run the main processing script:

```bash
python -m scripts.process_and_segment \
    --raw_dir data/raw \
    --audio_dir data/audio \
    --embeddings_dir data/embeddings
```

### Command-Line Arguments

- `--raw_dir`: (Required) The directory containing the raw audio files.

- `--audio_dir`: (Required) The output directory for the segmented audio utterances.

- `--embeddings_dir`: (Required) The output directory for the speaker embeddings.

- `--cut_from_cleaned`: (Optional) If specified, the final audio segments will be cut from the source-separated (cleaned) audio. By default, they are cut from the original audio.
