#!/bin/bash

REPO_ID="openai-community/gpt2"
DEST_ROOT="./Converter/models"
REPO_NAME="$(basename "$REPO_ID")"
REPO_DIR="$DEST_ROOT/$REPO_NAME"
TAR_NAME="hf_gpt2.tar"
TAR_PATH="$DEST_ROOT/$TAR_NAME"

log_progress() {
    echo "$(date +'%Y-%m-%d %H:%M:%S') - $1"
}

download_hf_repo() {
    log_progress "Preparing destination directory: $DEST_ROOT"
    mkdir -p "$DEST_ROOT"

    # Clean any previous contents for a fresh snapshot
    if [ -d "$REPO_DIR" ]; then
        log_progress "Removing existing directory: $REPO_DIR"
        rm -rf "$REPO_DIR"
    fi

    # Choose CLI: prefer 'hf', fallback to 'huggingface-cli'
    if command -v hf >/dev/null 2>&1; then
        DL_CMD=(hf download "$REPO_ID" --repo-type model --local-dir "$REPO_DIR" --local-dir-use-symlinks False)
    elif command -v huggingface-cli >/dev/null 2>&1; then
        DL_CMD=(huggingface-cli download "$REPO_ID" --repo-type model --local-dir "$REPO_DIR" --local-dir-use-symlinks False)
    else
        log_progress "Error: neither 'hf' nor 'huggingface-cli' is installed or on PATH."
        return 1
    fi

    log_progress "Downloading $REPO_ID to $REPO_DIR"
    if ! "${DL_CMD[@]}"; then
        log_progress "Error downloading $REPO_ID."
        return 1
    fi
}

remove_unnecessary_files() {
    log_progress "Removing unnecessary files from $REPO_DIR"

    # Remove file types
    if ! find "$REPO_DIR" -type f \( -name "*.tflite" -o -name "*.h5" -o -name "*.msgpack" -o -name "*.ot" \) -print -delete; then
        log_progress "Warning: some deletions may have failed (file extensions)."
    fi

    # Remove .gitattributes (file)
    rm -f "$REPO_DIR/.gitattributes" 2>/dev/null || true
    if ! find "$REPO_DIR" -type f -name ".gitattributes" -print -delete; then
        log_progress "Warning: deleting .gitattributes may have failed."
    fi

    # Remove any 'onnx' directories (top-level or nested)
    if ! find "$REPO_DIR" -type d -name "onnx" -prune -print -exec rm -rf {} +; then
        log_progress "Warning: removing onnx directories may have failed."
    fi

    # Remove any '.cache' directories (top-level or nested)
    if ! find "$REPO_DIR" -type d -name ".cache" -prune -print -exec rm -rf {} +; then
        log_progress "Warning: removing .cache directories may have failed."
    fi
}

tar_repo() {
    log_progress "Creating tarball: $TAR_PATH"
    rm -f "$TAR_PATH"

    # Create tar with folder as root entry (so it untars into gpt2/)
    if ( cd "$DEST_ROOT" && tar -cf "$TAR_NAME" "$REPO_NAME" ); then
        log_progress "Tarball created at $TAR_PATH"
    else
        log_progress "Error creating tarball."
        return 1
    fi
}

main() {
    log_progress "Starting Hugging Face repo fetch and packaging pipeline."
    download_hf_repo || { log_progress "Download step failed."; exit 1; }
    remove_unnecessary_files
    tar_repo || { log_progress "Tar step failed."; exit 1; }
    log_progress "All tasks completed successfully."
}

main "$@"
