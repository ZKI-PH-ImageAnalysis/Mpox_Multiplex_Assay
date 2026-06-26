#!/bin/bash
#SBATCH --job-name=MpoX-Classification-devBRANCH-newtestfile
#SBATCH --output=MpoX-Classification-devBRANCH-newtestfile.txt
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00
#SBATCH --partition=h100,h200,zki
#SBATCH --mem=80GB

set -euo pipefail

python main.py --input-file ../dataInputAll.csv \
 --test-file ../InstrumentVergleich_Input.csv \
 --outdir  2026_05_07-instrument \
 --filter ../Samples_Pre_MVA_Positive_SPox_2.csv
