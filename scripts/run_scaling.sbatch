#!/bin/bash
#SBATCH --job-name=knn-scale
#SBATCH --partition=haswell
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --output=logs/%x-%j.out

# Serial vs OpenMP (1,2,4,8,16 fire), aceeasi problema. Verifica ca metricile
# sunt identice cu cele seriale. Lansare din radacina proiectului:
#   sbatch scripts/run_scaling.sbatch [--ntrain N --nval N --ntest N]
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[ -s data/train-images-idx3-ubyte ] || bash scripts/download_mnist.sh data
make clean && make serial omp || exit 1
echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-1}"
mkdir -p build/out

./build/knn_serial "$@" > build/out/serial.txt
base=$(sed -n 's/.*search=\([0-9.]*\).*/\1/p' build/out/serial.txt)
sed -n '/== TEST ==/,/matrice/p;/^ *[0-9]: /p' build/out/serial.txt > build/out/serial.metrics
grep -E "accuracy|k ales" build/out/serial.txt | head -8
printf "\n%-10s %10s %9s %11s %s\n" impl search_s speedup efficiency metrici
printf "%-10s %10s %9s %11s %s\n" serial "$base" 1.00 - ref

for t in 1 2 4 8 16; do
  [ "$t" -le "${SLURM_CPUS_PER_TASK:-1}" ] || continue
  OMP_NUM_THREADS=$t OMP_PROC_BIND=close OMP_PLACES=cores ./build/knn_omp "$@" > build/out/omp$t.txt
  s=$(sed -n 's/.*search=\([0-9.]*\).*/\1/p' build/out/omp$t.txt)
  sed -n '/== TEST ==/,/matrice/p;/^ *[0-9]: /p' build/out/omp$t.txt > build/out/omp$t.metrics
  same=$(cmp -s build/out/serial.metrics build/out/omp$t.metrics && echo identice || echo DIFERITE)
  awk -v t=$t -v s=$s -v b=$base -v m=$same 'BEGIN{printf "%-10s %10.3f %9.2f %11.2f %s\n","omp-"t,s,b/s,b/s/t,m}'
done
