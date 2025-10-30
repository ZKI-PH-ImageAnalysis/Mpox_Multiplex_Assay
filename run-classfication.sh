#!/bin/bash
#SBATCH --job-name=MpoX-Classification
#SBATCH --output=MpoX-Classification.txt
#SBATCH --cpus-per-task=128
#SBATCH --time=72:00:00
#SBATCH --mem=80GB
#SBATCH --gres=local:30

python main.py --input-file ../dataInputAll.csv \
 --test-file ../dataPredictionRevisionRep.csv \
 --outdir  ../2025_06_16-GBC-baseline \
 --filter ../Samples_Pre_MVA_Positive_SPox_2.csv
