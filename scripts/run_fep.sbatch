#!/bin/bash
#SBATCH --job-name=knn
#SBATCH --partition=haswell
#SBATCH --account=asc
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x-%j.out

# Lansare (din radacina proiectului):  mkdir -p logs && sbatch scripts/run_fep.sbatch
# Argumente optionale pentru program:  sbatch scripts/run_fep.sbatch --ntrain 30000
# Alt compilator/modul:                MODULES="gcc/12" sbatch ...   (vezi `module avail`)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
echo "pwd=$(pwd)"; ls
for f in Makefile src/main.c src/knn_serial.c scripts/download_mnist.sh; do
  [ -e "$f" ] || { echo "EROARE: lipseste $f in $(pwd). Lanseaza sbatch din radacina proiectului copiat complet." >&2; exit 1; }
done

if command -v module >/dev/null 2>&1 && [ -n "${MODULES:-}" ]; then
  for m in $MODULES; do module load "$m"; done
fi

# Datele se descarca o singura data (daca nodul de calcul nu are internet,
# ruleaza inainte pe login node:  bash scripts/download_mnist.sh)
if [ ! -s data/train-images-idx3-ubyte ]; then
  bash scripts/download_mnist.sh data
fi

# -march=native pe nodul de calcul = Haswell (AVX2) => compileaza AICI, nu pe login.
command -v gcc >/dev/null || { echo "EROARE: gcc lipseste. Foloseste MODULES=\"<modul gcc>\" sbatch ... (vezi 'module avail')" >&2; exit 1; }
gcc --version | head -1
make clean && make serial

echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-1} start=$(date +%T)"
srun ./build/knn_serial "$@"
echo "end=$(date +%T)"
