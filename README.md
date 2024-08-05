# Mpox_Multiplex_Assay
Machine Learning based differentiation between Mpox infection and MVA immunization

## System Requirements 

### Software Dependencies
- Python >= 3.11.5
- Conda >= 23.3.1

### Operating Systems
Tested on Ubuntu 20.04.6

### Non-Standard Hardware
No specific hardware requirements; a standard desktop or laptop computer with at least 8 GB of RAM is recommended.
  

## Installation
**Clone the Repository**
```
git clone https://github.com/ZKI-PH-ImageAnalysis/Mpox_Multiplex_Assay.git
cd Mpox_Multiplex_Assay
```

**Create and Activate the Conda Environment**
```
conda env create -f environment.yml
conda activate Mpox_Classifier
```
_Typical install time on a "normal" desktop computer: 5-10 minutes._

## Quickstart

**Prepare the data**

A small, simulated dataset is provided in the data folder. The dataset file is sample_data.csv.

**Run the Demo or Your Own Analysis**
```
python main.py --input-file yourdata.csv --outdir results-dir/
```
For your own data, replace data/sample_data.csv with the path to your CSV file and demo_results with your desired output directory.

Ensure your data is in CSV format. Columns should match those in the provided sample dataset.

_Expected run time on a "normal" desktop computer: Approximately 5-10 minutes._

## Parameters
Detailed parameters and their descriptions can be found at the top of main_final.py.

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
