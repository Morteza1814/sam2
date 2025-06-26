#!/bin/bash
#SBATCH --job-name=download_sav
#SBATCH --output=logs/download_sav_%A_%a.out
#SBATCH --error=logs/download_sav_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G

# Set download destination
DEST_DIR="/bigtemp2/rgq5aw/sav_train/sav_tar"
mkdir -p "$DEST_DIR"

# Load GNU parallel if available (or skip if not)
module load parallel || true

# Define download function using wget
download_file() {
    name="$1"
    url="$2"
    echo "Downloading $name..."
    wget -O "$DEST_DIR/$name" "$url"
}
export -f download_file
export DEST_DIR

# Run parallel downloads
cat urls.txt | parallel --colsep '\t' -j 8 download_file {1} {2}
