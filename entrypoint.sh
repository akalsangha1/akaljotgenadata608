#!/bin/bash
set -e

# Download model artifacts from S3 then start Streamlit

ARTIFACT_DIR="${MODEL_ARTIFACT_DIR:-/app/model_artifacts}"
S3_BUCKET="${S3_BUCKET:-pulsepoint-raw-zone-akaljotmena}"
S3_PREFIX="${S3_MODEL_PREFIX:-model_artifacts}"

echo "[PulsePoint] Starting container..."

if [ ! -f "${ARTIFACT_DIR}/logistic_model.pkl" ]; then
    echo "[PulsePoint] Downloading model artifacts from s3://${S3_BUCKET}/${S3_PREFIX}/ ..."
    mkdir -p "${ARTIFACT_DIR}"
    aws s3 cp "s3://${S3_BUCKET}/${S3_PREFIX}/" "${ARTIFACT_DIR}/" --recursive
    echo "[PulsePoint] Artifacts downloaded:"
    ls -lh "${ARTIFACT_DIR}/"
else
    echo "[PulsePoint] Model artifacts already present at ${ARTIFACT_DIR}"
fi

echo "[PulsePoint] Launching Streamlit on port 8501..."
exec streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
