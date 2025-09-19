#!/bin/bash

# ==============================================================================
# Speech Dataset Factory - Full Pipeline Automation Script
# ==============================================================================
set -e # Exit immediately if a command exits with a non-zero status.
set -o pipefail # Return the exit status of the last command in the pipe that failed.

# --- Configuration ---
# --- You can modify these variables to fit your setup ---

# Project Directory
# Assumes the script is run from the root of the speech-dataset-factory directory
PROJECT_DIR=$(pwd)

# Data Directories
RAW_DIR="${PROJECT_DIR}/data/raw"
AUDIO_DIR="${PROJECT_DIR}/data/audio"
EMBEDDINGS_DIR="${PROJECT_DIR}/data/embeddings"
PRETRAINED_MODELS_DIR="${PROJECT_DIR}/pretrained_models"

# Manifest File Paths (the script will create these)
MANIFEST_BASE="data/manifest"
MANIFEST_INITIAL="${MANIFEST_BASE}.json"
MANIFEST_METRICS="${MANIFEST_BASE}_metrics.json"
MANIFEST_METRICS_FILTERED="${MANIFEST_BASE}_metrics_filtered.json"
MANIFEST_LANG="${MANIFEST_BASE}_lang.json"
MANIFEST_LANG_FILTERED="${MANIFEST_BASE}_lang_filtered.json"
MANIFEST_SPEAKERS="${MANIFEST_BASE}_speakers.json"
MANIFEST_NEMO="${MANIFEST_BASE}_transcribed_nemo.json"
MANIFEST_WHISPER="${MANIFEST_BASE}_transcribed_whisper.json"
MANIFEST_ASR_METRICS="${MANIFEST_BASE}_asr_metrics.json"
MANIFEST_FINAL="${MANIFEST_BASE}_final_filtered.json"

# Model Paths (replace with your actual model paths)
# For NeMo, this can be a path to a .nemo file or an NGC model name
NEMO_MODEL_PATH="nemo_path" # Example: Using a pretrained NGC model
# For Whisper, this should be the path to a faster-whisper compatible model directory
WHISPER_MODEL_PATH="whisper_path" # Example: Using a pretrained model name

# Pipeline Settings
MIN_SNR=25.0
MIN_C50=30.0
TARGET_LANG="fa" # Language to filter for (e.g., 'fa', 'en')
MIN_LANG_PROB=0.99
MAX_WER=0.15
MAX_CER=0.10
MIN_CHARS=3
CUT_FROM_CLEANED=true # Set to true to use source-separated audio for final cuts

# --- Helper Functions ---
log_step() {
    echo "=============================================================================="
    echo "➡️  STEP: $1"
    echo "=============================================================================="
}

# ==============================================================================
#                            🚀 START OF PIPELINE 🚀
# ==============================================================================

# --- Step 1: Installation and Setup ---
log_step "Installation and Setup"

# Download the pretrained source separation model
UVR_MODEL_PATH="${PRETRAINED_MODELS_DIR}/UVR-MDX-NET-Inst_HQ_3.onnx"
if [ ! -f "$UVR_MODEL_PATH" ]; then
    echo "⬇️  Downloading UVR source separation model..."
    wget https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_3.onnx -O "$UVR_MODEL_PATH"
    echo "✅ Model downloaded."
else
    echo "✅ UVR model already exists."
fi

# --- Step 2: Process and Segment Audio ---
log_step "Process and Segment Raw Audio"
echo "This step will standardize, denoise, diarize, and segment your raw audio files."
echo "Raw audio directory: ${RAW_DIR}"
echo "Segmented audio output: ${AUDIO_DIR}"
echo "Speaker embeddings output: ${EMBEDDINGS_DIR}"

CUT_FLAG=""
if [ "$CUT_FROM_CLEANED" = true ]; then
    CUT_FLAG="--cut_from_cleaned"
fi

python -m scripts.process_and_segment \
    --raw_dir "${RAW_DIR}" \
    --audio_dir "${AUDIO_DIR}" \
    --embeddings_dir "${EMBEDDINGS_DIR}" \
    ${CUT_FLAG}
echo "✅ Audio segmentation complete."


# --- Step 3: Create Initial Manifest ---
log_step "Create Initial Manifest"
python -m scripts.create_manifest \
    --audio_dir "${AUDIO_DIR}" \
    --embeddings_dir "${EMBEDDINGS_DIR}" \
    --manifest_path "${MANIFEST_INITIAL}"
echo "✅ Initial manifest created at ${MANIFEST_INITIAL}"


# --- Step 4: Calculate Quality Metrics and Filter ---
log_step "Calculate and Filter by Audio Quality Metrics (SNR & C50)"
python -m scripts.calculate_quality_metrics \
    --input_manifest "${MANIFEST_INITIAL}" \
    --output_manifest "${MANIFEST_METRICS}" \
    --overwrite

python -m scripts.post-processing.filter_low_quality_samples \
    --input_manifest "${MANIFEST_METRICS}" \
    --output_manifest "${MANIFEST_METRICS_FILTERED}" \
    --min_snr ${MIN_SNR} \
    --min_c50 ${MIN_C50}
echo "✅ Quality filtering complete. Filtered manifest at ${MANIFEST_METRICS_FILTERED}"


# --- Step 5: Detect Language and Filter ---
log_step "Detect and Filter by Language"
python -m scripts.detect_language \
    --input_manifest "${MANIFEST_METRICS_FILTERED}" \
    --output_manifest "${MANIFEST_LANG}" \
    --batch_size 2

python -m scripts.post-processing.filter_language \
    --input_manifest "${MANIFEST_LANG}" \
    --output_manifest "${MANIFEST_LANG_FILTERED}" \
    --lang "${TARGET_LANG}" \
    --min_prob ${MIN_LANG_PROB}
echo "✅ Language filtering complete. Filtered manifest at ${MANIFEST_LANG_FILTERED}"


# --- Step 6: Cluster Speakers for Unique IDs ---
log_step "Cluster Speakers to Get Unique IDs Across Files"
python -m scripts.cluster_speakers \
    --input_manifest "${MANIFEST_LANG_FILTERED}" \
    --output_manifest "${MANIFEST_SPEAKERS}"
echo "✅ Speaker clustering complete. Final speaker manifest at ${MANIFEST_SPEAKERS}"


# --- Step 7: Transcribe Audio (ASR) ---
log_step "Transcribe Audio with NeMo and Whisper"

# Transcribe with NeMo
echo "Running NeMo ASR..."
python -m scripts.asr \
    --model_type nemo \
    --model_path "${NEMO_MODEL_PATH}" \
    --input_manifest "${MANIFEST_SPEAKERS}" \
    --output_manifest "${MANIFEST_NEMO}" \
    --batch_size 32 \
    --save_iterations 64

# Transcribe with Whisper
echo "Running Whisper ASR..."
python -m scripts.asr \
    --model_type whisper \
    --model_path "${WHISPER_MODEL_PATH}" \
    --input_manifest "${MANIFEST_SPEAKERS}" \
    --output_manifest "${MANIFEST_WHISPER}" \
    --save_iterations 64
echo "✅ ASR transcription complete for both models."


# --- Step 8: Calculate ASR Metrics ---
log_step "Calculate ASR Metrics (WER/CER)"
echo "Using Whisper as ground truth and NeMo as predictions."
python -m scripts.calculate_asr_metrics \
    "${MANIFEST_WHISPER}" \
    "${MANIFEST_NEMO}" \
    --output_manifest "${MANIFEST_ASR_METRICS}"
echo "✅ ASR metrics calculated. Output at ${MANIFEST_ASR_METRICS}"


# --- Step 9: Filter Based on ASR Errors ---
log_step "Filter Manifest Based on ASR Error Rates and Text Length"
python -m scripts.post-processing.filter_asr_errors \
    --input_manifest "${MANIFEST_ASR_METRICS}" \
    --output_manifest "${MANIFEST_FINAL}" \
    --max_wer ${MAX_WER} \
    --max_cer ${MAX_CER} \
    --min_chars ${MIN_CHARS}
echo "✅ ASR error filtering complete."


# ==============================================================================
#                           🎉 PIPELINE COMPLETE 🎉
# ==============================================================================
echo ""
echo "All steps completed successfully!"
echo "Your final, cleaned, and transcribed dataset manifest is located at:"
echo "➡️  ${MANIFEST_FINAL}"
echo ""