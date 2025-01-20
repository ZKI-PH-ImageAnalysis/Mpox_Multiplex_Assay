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

_Expected runtime on a standard desktop computer: Approximately 20-45 minutes. You can reduce the size of the simulated data if the runtime is too long. Additionally, you can adjust the parameters for k-fold cross-validation to speed up the process. Consider using fewer folds by setting n_split to 2 and reducing the number of repetitions with reps to 1._

## Parameters
Detailed parameters and their descriptions can be found at the top of main.py.

## Output Directory

When you run the analysis, results are saved in the specified output directory. The directory will contain the following subdirectories and files:
- classified: Contains CSV files with predictions from each machine learning method used in the analysis. Each CSV file includes the results from the respective method and train test split.
- classified_with_threshold: Similar to the classified subdirectory, but with predictions that apply a threshold to LDA results.
- confusion_matrices: Contains plots of confusion matrices for the predictions. These plots visualize the performance of the classification models in terms of true positives, false positives, true negatives, and false negatives.
- feature_importance: Contains plots showing the importance of features for Random Forest (RF) and Gradient Boosting Tree models. These plots help in understanding which features contribute most to the predictions.
- metrics: Includes CSV files with performance metrics such as recall, precision, and F1 score for each evaluated algorithm. This section provides a quantitative assessment of the model performance.
- misclassified_data: Contains CSV files with samples that were misclassified by the models. This allows for an examination of where the models are making errors.
- rule_base: Provides the rules generated by the FRBC (Fuzzy Rule-Based Classifier) method. These rules explain how the FRBC method makes its predictions.
- unknown_SPox_preds: Contains predictions for the SPox panel where the ground truth was not available. This section provides insights into the model’s performance on data with unknown labels.


## Set of algorithms
The following algorithms are evaluated:
- LDA
- LDA with Threshold
- LDA + RF
- RF
- LDA + FRBC
- FRBC
- FRBC with Threshold
- GradientBoostingClassifier
