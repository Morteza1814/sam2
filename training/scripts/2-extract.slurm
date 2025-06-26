#!/bin/bash
#SBATCH --job-name=untar_sav
#SBATCH --output=logs/untar_sav_extract.out
#SBATCH --error=logs/untar_sav_extract.out
#SBATCH --time=4:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G

# Set the directory containing the .tar files
TAR_DIR="/bigtemp2/rgq5aw/sav_train/sav_tar/"

# Set the destination directory for extracted contents
OUT_DIR="/bigtemp2/rgq5aw/sav_train/sav_data/"
mkdir -p "$OUT_DIR"

# Max parallel jobs
MAX_JOBS=8
JOB_COUNT=0

for file in "$TAR_DIR"/*.tar; do
    echo "Extracting $file ..."
    tar -xvf "$file" -C "$OUT_DIR" &
    ((JOB_COUNT++))

    if [[ $JOB_COUNT -ge $MAX_JOBS ]]; then
        wait
        JOB_COUNT=0
    fi
done

# Wait for any remaining background jobs to finish
wait
