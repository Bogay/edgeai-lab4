#!/bin/bash

# Configuration
INPUT_DIR="audio"           # Directory containing your original .wav files
OUTPUT_DIR="audio_processed" # Directory where processed files will be saved
TARGET_SR=16000             # Target sample rate (Hz)
TARGET_CHANNELS=1           # Target number of audio channels (1 for mono)

echo "--- Starting Audio Processing ---"

# --- 1. Check if FFmpeg is installed ---
if ! command -v ffmpeg &> /dev/null
then
    echo "Error: ffmpeg command not found."
    echo "Please ensure FFmpeg is installed and accessible in your system's PATH."
    exit 1
fi

# --- 2. Check if input directory exists ---
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist."
    exit 1
fi

# --- 3. Create output directory if it doesn't exist ---
mkdir -p "$OUTPUT_DIR"

# --- 4. Find all .wav files ---
# Using `find` to handle subdirectories as well, if needed.
# For just top-level, a simple `*.wav` glob works too.
# files=($(find "$INPUT_DIR" -maxdepth 1 -type f -name "*.wav"))
files=("$INPUT_DIR"/*.wav)

# Check if any .wav files were found
if [ "${#files[@]}" -eq 0 ] || [ ! -e "${files[0]}" ]; then
    echo "No .wav files found in '$INPUT_DIR'."
    exit 0
fi

TOTAL_FILES=${#files[@]}
PROCESSED_COUNT=0

echo "Processing $TOTAL_FILES .wav files from '$INPUT_DIR'..."
echo "Outputting to '$OUTPUT_DIR' (Target: ${TARGET_SR}Hz, ${TARGET_CHANNELS} channel(s))."

# --- 5. Process each .wav file ---
for input_filepath in "${files[@]}"; do
    # Skip if the glob pattern matched literally (e.g., if no files found)
    if [ ! -f "$input_filepath" ]; then
        continue
    fi

    filename=$(basename "$input_filepath")
    output_filepath="$OUTPUT_DIR/$filename"

    PROCESSED_COUNT=$((PROCESSED_COUNT + 1))
    echo "[$PROCESSED_COUNT/$TOTAL_FILES] Processing '$filename'..."

    # FFmpeg command:
    # -i: input file
    # -ar: set audio sample rate
    # -ac: set number of audio channels
    # -c:a pcm_s16le: set audio codec to uncompressed 16-bit PCM little-endian
    # -y: overwrite output files without asking
    # -hide_banner -loglevel error: suppress FFmpeg's verbose output, only show errors
    ffmpeg -i "$input_filepath" \
           -ar "$TARGET_SR" \
           -ac "$TARGET_CHANNELS" \
           -c:a pcm_s16le \
           -y \
           -hide_banner -loglevel error \
           "$output_filepath"

    # Check FFmpeg's exit status
    if [ $? -ne 0 ]; then
        echo "  Error: FFmpeg failed to process '$filename'. Skipping."
    fi
done

echo "--- Audio Processing Complete! ---"
echo "Processed files are saved in: '$OUTPUT_DIR'"
echo ""

