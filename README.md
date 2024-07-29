# Mpox_Multiplex_Assay
Machine Learning based differentiation between Mpox infection and MVA immunization


## Installation
```
conda create -n Mpox_Classifier -f environment.yml
conda activate Mpox_Classifier
```

## Quickstart
```
python main_final.py --input-file ../yourdata.csv --outdir ../results
```

## Parameters
Parameters can be found at top of main_final.py 

## Set of algorithms
The following algorithms are evaluated:
- LDA
- LDA with Threshold
- LDA + RF
- RF
- LDA + FRBC
- FRBC
- FRBC with Threshold
- XGBoost
