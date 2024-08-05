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
_Typical install time on a standard desktop computer: 5-10 minutes._

## Quickstart

**Prepare the data**

A script to generate a simulated dataset is provided. Adjust the number of rows and other parameters in the script as needed. To generate simulated data, run:
```
python simulate-data.py
```
This will create a CSV file named simulated_data.csv.

**Run the Demo or Your Own Analysis**
```
python main.py --input-file simulated_data.csv --outdir results-dir/
```
For your own data, replace simulated_data.csv with the path to your CSV file and results-dir/ with your desired output directory.

Ensure your data is in CSV format and the columns match those in the provided sample dataset.

If your input CSV is not preprocessed, add the `--preprocessed-input False` parameter.

_Expected runtime on a standard desktop computer: Approximately 10-20 minutes. You can reduce the size of the simulated data if the runtime is too long._

## Parameters
Detailed parameters and their descriptions can be found at the top of main.py.

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
