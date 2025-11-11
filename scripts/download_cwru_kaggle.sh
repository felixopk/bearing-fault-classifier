#!/bin/bash

# ======================================================================
#  CWRU Bearing Dataset Downloader (Kaggle)
# ======================================================================

echo "=========================================================================="
echo "  CWRU Bearing Dataset Download (Kaggle)"
echo "=========================================================================="
echo ""

# Check if Kaggle CLI is installed
if ! command -v kaggle &> /dev/null; then
    echo "⚠️  Kaggle CLI not installed. Installing..."
    pip install kaggle
fi

echo "📋 Setup Instructions:"
echo ""
echo "1. Go to: https://www.kaggle.com/settings/account"
echo "2. Scroll to 'API' section"
echo "3. Click 'Create New API Token'"
echo "4. Download kaggle.json file"
echo "5. Move it to: ~/.kaggle/kaggle.json"
echo "   (Linux/Mac: mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/)"
echo "   (Windows: mkdir %HOMEPATH%\\.kaggle && move %HOMEPATH%\\Downloads\\kaggle.json %HOMEPATH%\\.kaggle\\)"
echo "6. Set permissions: chmod 600 ~/.kaggle/kaggle.json"
echo ""
read -p "Press Enter after completing setup..."

# Create directories
mkdir -p /home/user/DevOps/projects/bearing-fault-classifier/data/raw
cd data/raw || exit

# Download dataset
echo ""
echo "📥 Downloading CWRU dataset from Kaggle..."
kaggle datasets download -d brjapon/cwru-bearing-datasets

# Unzip
echo "📦 Extracting files..."
unzip -o -q cwru-bearing-datasets.zip


