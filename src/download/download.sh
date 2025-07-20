#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --account=m1266
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=drivaerml_dl
#SBATCH --output=download_%j.log
#SBATCH --error=download_%j.err

# ----------------------------
# CONFIGURATION
# ----------------------------

HF_OWNER="neashton"
HF_PREFIX="drivaerml"
LOCAL_DIR="/pscratch/sd/j/jwl50/drivaerml/.cache"

# Token from argument or environment
if [[ -n "$1" ]]; then
    HF_TOKEN="$1"
elif [[ -n "$HF_TOKEN" ]]; then
    HF_TOKEN="$HF_TOKEN"
else
    echo "Error: Hugging Face token not provided. Pass it as a command line arg or set HF_TOKEN env variable."
    exit 1
fi

mkdir -p "$LOCAL_DIR"

# ----------------------------
# DOWNLOAD FUNCTION
# ----------------------------

download_single_run() {
    i="$1"
    RUN_DIR="run_$i"
    RUN_LOCAL_DIR="$LOCAL_DIR/$RUN_DIR"
    mkdir -p "$RUN_LOCAL_DIR"

    echo "Downloading run_$i..."

    wget --header="Authorization: Bearer $HF_TOKEN" \
        "https://huggingface.co/datasets/${HF_OWNER}/${HF_PREFIX}/resolve/main/$RUN_DIR/drivaer_${i}.stl" \
        -O "$RUN_LOCAL_DIR/drivaer_${i}.stl"

    wget --header="Authorization: Bearer $HF_TOKEN" \
        "https://huggingface.co/datasets/${HF_OWNER}/${HF_PREFIX}/resolve/main/$RUN_DIR/boundary_${i}.vtp" \
        -O "$RUN_LOCAL_DIR/boundary_${i}.vtp"

    wget --header="Authorization: Bearer $HF_TOKEN" \
        "https://huggingface.co/datasets/${HF_OWNER}/${HF_PREFIX}/resolve/main/$RUN_DIR/geo_ref_${i}.csv" \
        -O "$RUN_LOCAL_DIR/geo_ref_${i}.csv"
}

export -f download_single_run

# ----------------------------
# MAIN: PARALLEL EXECUTION
# ----------------------------

# Load parallel if needed
module load parallel

# Use GNU parallel to run multiple downloads in parallel
seq 1 500 | parallel -j $SLURM_CPUS_PER_TASK download_single_run
