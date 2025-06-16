#!/bin/bash

mkdir -p artifacts

# URL dari model dan preprocessor
MODEL_URL="http://13.214.202.57/similarity_model.h5"
PREPROCESSOR_URL="http://13.214.202.57/preprocessor.joblib"

# Unduh hanya jika belum ada
if [ ! -f "artifacts/similarity_model.h5" ]; then
    echo "Downloading model..."
    curl -o artifacts/similarity_model.h5 $MODEL_URL
fi

if [ ! -f "artifacts/preprocessor.joblib" ]; then
    echo "Downloading preprocessor..."
    curl -o artifacts/preprocessor.joblib $PREPROCESSOR_URL
fi
