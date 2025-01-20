import click
import pathlib

import numpy as np
import pandas as pd

from classifiers import *

import os
import math
import itertools
from sklearn.model_selection import StratifiedKFold

# algorithm_list = ["LDA", "RF", "XGBoost", "LDA_Threshold", "LDA_RF"]
antibody_list = ["IgM_IgG", "IgG"]
sero_list = ["all"]
data_cols_list = [ "dataIn"]
exclusion_list = [("L1R", "M1", "VACV")] 
# k for k-fold cross validation
n_split = 5
# How often we want to repeat k-fold cross validation 
reps = 3
seed_runs = reps*n_split
start_seed = 70


    
def group_df_analyte(df, data_column="data"):
    df = df.pivot_table(
        values=data_column,
        # index=["sampleID_metadata", "panel_detail", "panel", "serostatus_delta_IgG"],
        index=["sampleID_metadata", "panel_detail", "panel"],
        columns=["analyte"],
        aggfunc="first",
        dropna=True,
    )

    df = df.dropna()

    df = df.reset_index(level=["panel_detail"])

    return df


def process_for_skmoefs(
    df, 
    train_df, #added new parameter -> min and max of training data will be used for normalization
    is_spox,
    out="df_igg_all_panel.dat",
    panel_name="df_igg_all_panel",
):
    #frbc_labels -> removed as it wasn't used
    print("spox True or False:", is_spox)
    if is_spox == False:
        unique = df["panel_detail"].unique()
        unique_counter = df["panel_detail"].value_counts(sort=False).values
        for i in range(len(unique)):
            if unique_counter[i] == 1:
                new = df.loc[df["panel_detail"] == unique[i]]
                df = df.append(new)
    else:
        #print('is_spox columns\n', df.columns)
        unique = df["panel_detail"].unique()
        unique_counter = df["panel_detail"].value_counts(sort=False).values
        for i in range(len(unique)):
            if unique_counter[i] == 1:
                new = df.loc[df["panel_detail"] == unique[i]]
                df = df.append(new)

    cols = list(df.columns)

    if is_spox == False:
        last_col = cols[-1]
        a, b = cols.index("panel_detail"), cols.index(last_col)
        cols[b], cols[a] = cols[a], cols[b]
        df = df[cols]

        df.replace({"panel_detail": "MPXV"}, {"panel_detail": 0}, inplace=True)
    
        df.replace({"panel_detail": "MVA"}, {"panel_detail": 1}, inplace=True)
    
        df.replace({"panel_detail": "Pre"}, {"panel_detail": 2}, inplace=True)
        
        df.replace({"panel_detail": "CPXV"}, {"panel_detail": 3}, inplace=True)
    else:
        last_col = cols[-1]
        a, b = cols.index("panel_detail"), cols.index(last_col)
        cols[b], cols[a] = cols[a], cols[b]
        df = df[cols]

        df.replace({"panel": "MPXV"}, {"panel_detail": 0}, inplace=True)
    
        df.replace({"panel": "MVA"}, {"panel_detail": 1}, inplace=True)
    
        df.replace({"panel": "Pre"}, {"panel_detail": 2}, inplace=True)
        
        df.replace({"panel": "CPXV"}, {"panel_detail": 3}, inplace=True)

    print('skmoefs\n', df)

    data_np = df.to_numpy()

    minvalue_series = train_df.min(axis=0)
    maxvalue_series = train_df.max(axis=0)

    f = open(out, "w")
    f.write(f"@relation {panel_name}")
    f.write("\n")

    cols = list(df.columns)
    
    last = len(cols) - 1
    #if is_spox == True:
    #    last = len(cols)
    for i in range(last):
        f.write("@attribute ")
        f.write(str(cols[i]))
        f.write(" real [")
        f.write(str(minvalue_series[i]))
        f.write(", ")
        f.write(str(maxvalue_series[i]))
        f.write("]")
        f.write("\n")

    f.write("@inputs ")
    for i in range(last):
        f.write(str(cols[i]))
        if i != last - 1:
            f.write(", ")
    f.write("\n")

    f.write("@outputs ")
    f.write("panel_detail")
    f.write("\n")

    f.write("@data")
    f.write("\n")

    for i in range(data_np.shape[0]):
        for j in range(data_np.shape[1]):
            f.write(str(data_np[i][j]))
            if j != data_np.shape[1] - 1:
                f.write(", ")
        f.write("\n")
    f.close()


def preprocess_spox(
    input_file,
    filter_csv,
    antibody="IgG",
    sero_th="all",
    data_column="dataIn",
    exclude_features=("None"),
    preprocessed=False,
):
    """
    Antibody: "IgG", "IgM", "IgM_IgG"
    sero_th: "all", "positive", "borderline positive"
    data_column: "data", "dataln"
    panel: "all", "acute", "epi"
    exclude_features: list of features we would like to exclude for example ["M1", "L1R"]
    """
    df_in = pd.read_csv(input_file, low_memory=False)
    df_in = df_in.dropna()
    
    # Replace -inf with NaN
    df_in.replace([np.inf, -np.inf], np.nan, inplace=True)


    # dataIn columns
    dataIn_columns = [col for col in df_in.columns if col.startswith('dataIn')]
    columns_to_keep = ['sampleID_meta', 'serostatus_cat.delta', 'panel']

    # reshape
    def transform_row(row):
        new_data = {}
        for col in dataIn_columns:
            isotype = row['isotype']
            base_name = col.split('_', 1)[1]  # Extract the base name after 'dataIn_'
            new_column_name = f"{isotype}_{base_name}"
            new_data[new_column_name] = row[col]
        return new_data
    transformed_rows = df_in.apply(transform_row, axis=1).apply(pd.Series)
    df_in = pd.concat([df_in[columns_to_keep], transformed_rows], axis=1)
    # drop duplicates
    df_in = df_in.groupby('sampleID_meta', as_index=False).first()

    # Fow now filter antibody values
    if antibody == "IgM_IgG":
    # Keep all columns
        df_in = df_in
    else:
        # Choose only "IgG" or "IgM" columns
        antibody_columns = [col for col in df_in.columns if col.startswith(antibody)]
        df_in = df_in[['sampleID_meta', 'serostatus_cat.delta', 'panel'] + antibody_columns]
    
    # Add column for explicit serostatus of delta antigen
    df_in["serostatus_delta_IgG"] = df_in.apply(lambda x: x['serostatus_cat.delta'], axis=1)
    serostatus_IDs = df_in[df_in["serostatus_delta_IgG"].notna()]

    if sero_th == "positive":
        serostatus_IDs = serostatus_IDs[serostatus_IDs["serostatus_delta_IgG"].isin(["positive"])]
    elif sero_th == "borderline positive":
        serostatus_IDs = serostatus_IDs[
            serostatus_IDs["serostatus_delta_IgG"].isin(["borderline positive", "positive"])
        ]
    serostatus_IDs = serostatus_IDs["sampleID_meta"].unique()

    print(
        f"ATTENTION: Dataframe includes {df_in.panel.isna().sum()} rows with NaN values in panel_detail.\
        These will be excluded from further analysis."
    )
    # Drop NaN
    df_in = df_in[df_in["panel"].notna()]
    #df_in = df_in[df_in["dataIn_D8L"].notna()]

    # Group by patient ID so that we have analytes as columns
    df_out = df_in

    # Drop column if in exclude_features
    # need to use endswith so it will work with IgM+IgG data
    # where columns look like this "IgM_M1", "IgG_M1", ..
    cols_to_drop = df_out.columns[df_out.columns.str.endswith(exclude_features)]
    df_out = df_out.drop(cols_to_drop, axis=1, errors="ignore")

    # Filter out IDs from csv
    if filter_csv is not None and os.path.isfile(filter_csv):
        df_filter = pd.read_csv(filter_csv, low_memory=False)
        df_filter = df_filter.rename(columns={"excludeIDs": "sampleID_meta"})
        print(f"Filtering: {len(df_filter)} samples were removed from analysis.")
        df_joined = df_out.merge(df_filter, on='sampleID_meta', how="inner", indicator=True).drop("_merge", axis=1)
        df_out = df_out.merge(df_filter, on='sampleID_meta', how="outer", indicator=True)
        df_out = df_out[df_out['_merge'] == 'left_only'].drop("_merge", axis=1)

    # Filter only for the serostatus of delta IgG
    if not sero_th == "all":
        df_out = df_out[df_out['sampleID_meta'].isin(serostatus_IDs)]
            
    # Reset index
    df_out = df_out.set_index(["sampleID_meta"])

    # Remove CPXV for now
    df_out = df_out[df_out["panel"] != "CPXV"]
    
    # Replace Pre_New samples with Pre
    df_out.loc[df_out.panel == "Pre_New", 'panel'] = "Pre"

    # Remove the Spox and Spox_Rep columns
    df_out = df_out[df_out["panel"] != "SPox"]
    df_out = df_out[df_out["panel"] != "SPox_Rep"]

    # Panel detail
    df_out["panel_detail"] = df_out["panel"]

    df_all = df_out.drop(["serostatus_cat.delta", "serostatus_delta_IgG", "panel"], axis=1)
    
    return df_all


def preprocess_data(
    df_in,
    test_file,
    filter_csv,
    antibody="IgG",
    sero_th="all",
    data_column="data",
    exclude_features=("None"),
    preprocessed=False,
):
    """
    Antibody: "IgG", "IgM", "IgM_IgG"
    sero_th: "all", "positive", "borderline positive"
    data_column: "data", "dataln"
    panel: "all", "acute", "epi"
    exclude_features: list of features we would like to exclude for example ["M1", "L1R"]
    """
    if preprocessed:
        df_out = df_in
    else:
        # Replace -inf with NaN
        df_in.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Fow now filter antibody values
        if antibody == "IgM_IgG":
            # Filter out IgA or other analytes
            df_in = df_in[df_in["isotype"].isin(["IgM", "IgG"])]
            df_in["analyte"] = df_in["isotype"] + "_" + df_in["analyte"]
            #df_in["analyte"] = "dataIn_" + df_in["analyte"]
        else:
            df_in = df_in[df_in["isotype"] == antibody]
            df_in["analyte"] = df_in["isotype"] + "_" + df_in["analyte"]
            #df_in["analyte"] = "dataIn_" + df_in["analyte"]

        # Only select necessary columns, for now Analyte(s)
        df_in = df_in[
            [
                "sampleID_metadata",
                "panel_detail",
                "panel",
                "analyte",
                data_column,
                "serostatus_cat.delta",
            ]
        ]

        # Add column for explicit serostatus of delta antigen
        df_in["serostatus_delta_IgG"] = df_in.apply(lambda x: x['serostatus_cat.delta'] if x["analyte"] == "IgG_Delta" else np.nan, axis=1)
        serostatus_IDs = df_in[df_in["serostatus_delta_IgG"].notna()]

        if sero_th == "positive":
            serostatus_IDs = serostatus_IDs[serostatus_IDs["serostatus_delta_IgG"].isin(["positive"])]
        elif sero_th == "borderline positive":
            serostatus_IDs = serostatus_IDs[
                serostatus_IDs["serostatus_delta_IgG"].isin(["borderline positive", "positive"])
            ]
        serostatus_IDs = serostatus_IDs["sampleID_metadata"].unique()

        print(
            f"ATTENTION: Dataframe includes {df_in.panel_detail.isna().sum()} rows with NaN values in panel_detail.\
            These will be excluded from further analysis."
        )

        # Drop NaN
        df_in = df_in[df_in["panel_detail"].notna()]

        # Group by patient ID so that we have analytes as columns
        df_out = group_df_analyte(df_in, data_column=data_column)

        # Drop column if in exclude_features
        # need to use endswith so it will work with IgM+IgG data
        # where columns look like this "IgM_M1", "IgG_M1", ..
        cols_to_drop = df_out.columns[df_out.columns.str.endswith(exclude_features)]
        df_out = df_out.drop(cols_to_drop, axis=1, errors="ignore")

        # Convert multi-index to columns
        df_out = df_out.reset_index()
        
        # Filter out IDs from csv
        if filter_csv is not None and os.path.isfile(filter_csv):
            df_filter = pd.read_csv(filter_csv, low_memory=False)
            df_filter = df_filter.rename(columns={"excludeIDs": "sampleID_metadata"})
            print(f"Filtering: {len(df_filter)} samples were removed from analysis.")
            df_joined = df_out.merge(df_filter, on='sampleID_metadata', how="inner", indicator=True).drop("_merge", axis=1)
            df_out = df_out.merge(df_filter, on='sampleID_metadata', how="outer", indicator=True)
            df_out = df_out[df_out['_merge'] == 'left_only'].drop("_merge", axis=1)

        # Filter only for the serostatus of delta IgG
        if not sero_th == "all":
            df_out = df_out[df_out['sampleID_metadata'].isin(serostatus_IDs)]
    
    # Reset index
    df_out = df_out.set_index(["sampleID_metadata"])

    # drop this again
    # df_out = df_out.drop("serostatus_delta_IgG", axis=1)

    # Remove CPXV for now
    df_out = df_out[df_out["panel_detail"] != "CPXV"]

    # Extract the unknown samples as df_spox
    #df_spox = df_out[df_out["panel_detail"] == "SPox"].drop(["panel"], axis=1)

    # Join the filtered out samples to the Spox dataframe
    #if filter_csv is not None and os.path.isfile(filter_csv):
    #    df_joined = df_joined.drop("panel", axis=1)
    #    df_joined = df_joined.set_index(["sampleID_metadata"])
    #    df_spox = pd.concat([df_spox, df_joined])


    # extract the repetition panel
    df_rep = df_out[df_out["panel_detail"] == "SPox_Rep"].drop(["panel"], axis=1)
    # Add -rep to ID
    df_rep = df_rep.rename(index=lambda s: s + '-rep')
    # concat them both
    #df_spox = pd.concat([df_spox, df_rep])
    df_spox = preprocess_spox(
            test_file,
            filter_csv,
            antibody,
            sero_th,
            data_column,
            exclude_features,
            preprocessed
        )
    # Specify columns to retain at the front
    columns_to_keep = ['panel_detail']
    # Get all other columns (excluding 'panel')
    other_columns = [col for col in df_spox.columns if col != 'panel_detail']
    # Order all columns alphabetically, starting with 'panel'
    columns_order = columns_to_keep + sorted(other_columns)
    # Reorder the DataFrame
    df_spox = df_spox[columns_order]
    
    # Replace Pre_New samples with Pre
    df_out.loc[df_out.panel_detail == "Pre_New", 'panel_detail'] = "Pre"

    # Remove the Spox and Spox_Rep columns
    df_out = df_out[df_out["panel_detail"] != "SPox"]
    df_out = df_out[df_out["panel_detail"] != "SPox_Rep"]
    
    # Order all columns, start with panel detail, and then all other columns 
    columns_to_keep = ['panel_detail']
    # Get all other columns (excluding 'panel')
    other_columns = [col for col in df_out.columns if col != 'panel_detail']
    # Order all columns alphabetically, starting with 'panel'
    columns_order = columns_to_keep + sorted(other_columns)
    # Reorder the DataFrame
    df_out = df_out[columns_order]

    # Split to the three panels and drop the panel column, not needed anymore
    df_all = df_out.drop(["panel"], axis=1)
    df_acute = df_out[df_out["panel"] != "SPox"].drop(["panel"], axis=1)
    df_epi = df_out[df_out["panel"] == "SPox"].drop(["panel"], axis=1)


    print('df_all\n', df_all)
    print('df_spox\n', df_spox)
    # Compare these two dataframes and make sure they columns are ordered equally
    if list(df_all.columns) == list(df_spox.columns):
        print("The column order is identical.")
    else:
        print("The column order is different.")

    return df_all, df_acute, df_epi, df_spox
    
    
def set_split(df_train, df_test, seed, n_split=5):
    # If identical we use train test split
    if df_train.columns.name  == df_test.columns.name :
        X = df_train.iloc[:, 1:]
        y = df_train.iloc[:, 0]
        # dataset split depends on n_split and current seed 
        # for performing repeated k-fold 
        kfold_seed = math.floor(seed / n_split)
        skfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=kfold_seed)

        # because we choose our seeds from 70 to 80
        k = seed % n_split
        train_idx, test_idx = list(skfold.split(X,y))[k]

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        
    # Otherwise we use complete dataset
    else:
        # Make sure there is no overlap, remove any potential overlap
        df_train = df_train.reset_index()
        df_test = df_test.reset_index()
        cond = df_train['sampleID_metadata'].isin(df_test['sampleID_metadata'])
        df_train = df_train.drop(df_train[cond].index)
        df_train = df_train.set_index('sampleID_metadata')
        df_test = df_test.set_index('sampleID_metadata')


        X_train = df_train.iloc[:, 1:]
        y_train = df_train.iloc[:, 0]
        X_test = df_test.iloc[:, 1:]
        y_test = df_test.iloc[:, 0]
        
    cont = None
    if not set(y_train.unique()) == set(y_test.unique()):
        cont = True
        
    return X_train, y_train, X_test, y_test, cont


@click.command()
@click.option(
    "--input-file",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path
    ),
    help = "Path to dataInput.csv",
    default = "dataInputAll.csv",
)
@click.option(
    "--test-file",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path
    ),
    help = "Path to additional dataInput.csv",
    default = "dataInputAll.csv",
)
@click.option(
    "--filter",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path
    ),
    help = "Path to a csv which includes samples we want to remove from the dataset",
    default = None,
)
@click.option(
    "--outdir",
    type=click.Path(
        file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
    help = "Path to results directory",
    default = ".../results_tmp/",
)
@click.option(
    "--preprocessed-input",
    type=bool,
    help = "Bool value if inputfile csv is already preprocessed",
    default = False,
)
def main(input_file, test_file, filter, outdir, preprocessed_input):
    df_assay = pd.read_csv(input_file, low_memory=False)

    d = {}
    for antibody, sero_status, data_col, exclude_cols in itertools.product(
        antibody_list, sero_list, data_cols_list, exclusion_list
    ):
        df_all, df_acute, df_epi, df_spox = preprocess_data(
            df_assay,
            test_file,
            filter,
            antibody=antibody,
            sero_th=sero_status,
            data_column=data_col,
            exclude_features=exclude_cols,
            preprocessed=preprocessed_input,
        )
        # just for readability when saving files
        exclude_cols = "".join(exclude_cols)
        d[
            f"antibody_{antibody}_serostatus_{sero_status}_datacol_{data_col}_excluding_{exclude_cols}"
        ] = [df_all, df_acute, df_epi, df_spox]

    end_seed = start_seed + seed_runs
    seeds = list(range(start_seed, end_seed))
    
    rule_folder = os.path.join(outdir, "rule_base/")
    if os.path.exists(rule_folder) == False:
        os.makedirs(rule_folder)
    metrics_folder = os.path.join(outdir, "metrics/")
    stat_folder = os.path.join(outdir, "statistical_data/")
    stat_folder_spox = os.path.join(outdir, "statistical_revised_data/")
    cm_folder = os.path.join(outdir, "confusion_matrices/")
    mis_folder = os.path.join(outdir, "misclassified_data/")
    class_threshold_folder = os.path.join(outdir, "classified_with_threshold/")
    classified_folder = os.path.join(outdir, "classified_general/")
    unknown_pred_folder = os.path.join(outdir, "unknown-Spox-preds/")
    LDA_folder = os.path.join(outdir, "LDA-plots/")
    feature_folder = os.path.join(outdir, "feature_importance/")

    # for dataframe in dataframe_list
    for df_name, [df_all, df_acute, df_epi, df_spox] in d.items():
        df_all.name = "all"
        df_acute.name = "acute"
        df_epi.name = "epi"
        df_spox.name = "spox"
        panel_l = [p for p in itertools.product([df_all, df_acute, df_epi], repeat=2)]

        # n_algs = len(algorithm_list)
        # TODO Automatically choose size of the following arrays

        precision = np.zeros((len(panel_l), 10, len(seeds)))
        accuracy = np.zeros((len(panel_l), 10, len(seeds)))
        recall = np.zeros((len(panel_l), 10, len(seeds)))
        f1 = np.zeros((len(panel_l), 10, len(seeds)))
        
        precision_spox = np.zeros((len(panel_l), 10, len(seeds)))
        accuracy_spox = np.zeros((len(panel_l), 10, len(seeds)))
        recall_spox = np.zeros((len(panel_l), 10, len(seeds)))
        f1_spox = np.zeros((len(panel_l), 10, len(seeds)))

        # Loop for all panel combiniation         
        panel_l = [p for p in itertools.product([df_all, df_acute, df_epi], repeat=2)]

        frbc_params = frbc_get_params(df_name)
        lda_frbc_params = lda_frbc_get_params(df_name)

        for run in range(len(seeds)):
            # Give them names so we know which df is currently used in the following loop
            df_all.columns.name  = "all"
            df_acute.columns.name  = "acute"
            df_epi.columns.name  = "epi"
            df_spox.columns.name  = "spox"

            # Loop to test across all combination of panels
            panel_l = [p for p in itertools.product([df_all, df_acute, df_epi], repeat=2)]
            for idx_panel, (df_train, df_test) in enumerate(panel_l):
                unique_train = df_train["panel_detail"].unique()
                unique_train_counter = df_train["panel_detail"].value_counts(sort=False).values
                unique_test = df_test["panel_detail"].unique()
                unique_test_counter = df_test["panel_detail"].value_counts(sort=False).values

                for i in range(len(unique_train)):
                    if unique_train_counter[i] == 1:
                        new = df_train.loc[df_train["panel_detail"] == unique_train[i]]
                        df_train = df_train.append(new)

                for i in range(len(unique_test)):
                    if unique_test_counter[i] == 1:
                        new = df_test.loc[df_test["panel_detail"] == unique_test[i]]
                        df_test = df_test.append(new)
                
                print('df name', df_train.name)
                X_train, y_train, X_test, y_test, cont = set_split(df_train, df_test, seeds[run], n_split)
                     
                if cont == True:
                    continue
        
                train_set_frbc = X_train
                train_set_frbc["panel_detail"] = y_train
                test_set_frbc = X_test
                test_set_frbc["panel_detail"] = y_test
                
                X_train, y_train, X_test, y_test, cont = set_split(df_train, df_test, seeds[run], n_split)
                
                df_spox = preprocess_spox(
                    test_file,
                    filter,
                    antibody=antibody,
                    sero_th=sero_status,
                    data_column=data_col,
                    exclude_features=exclude_cols,
                    preprocessed=preprocessed_input
                )
                                
                print('new spox\n', df_spox)
                
                df_name_with_panel = (
                    f"train_{df_train.columns.name }_test_{df_test.columns.name }_{df_name}"
                )
                train_sets = (X_train, y_train, X_test, y_test)

                process_for_skmoefs(
                    test_set_frbc,
                    train_set_frbc, #new parameter for normalization
                    False,
                    out=f"skmoefs/dataset/{df_name_with_panel}-TEST.dat",
                    panel_name=f"{df_name_with_panel}",
                )
                process_for_skmoefs(
                    train_set_frbc,
                    train_set_frbc, #new parameter for normalization
                    False,
                    out=f"skmoefs/dataset/{df_name_with_panel}-TRAIN.dat",
                    panel_name=f"{df_name_with_panel}",
                )
                train_sets_frbc = (train_set_frbc, test_set_frbc)
                
                #create file with spox data for frbc and lda_frbc
                #spox_frbc = df_spox.iloc[:, 1:]
                spox_frbc = df_spox
                process_for_skmoefs(
                    spox_frbc,
                    train_set_frbc, #new parameter for normalization
                    True,
                    out=f"skmoefs/dataset/{df_name_with_panel}-SPOX.dat",
                    panel_name=f"{df_name_with_panel}",
                )
                spox_set_frbc = spox_frbc

                # Test if number of classes at least 3

                if len(y_train.unique()) > 2:
                    accuracy[idx_panel][0][run], precision[idx_panel][0][run], recall[idx_panel][0][run], f1[idx_panel][0][run], accuracy_spox[idx_panel][0][run], precision_spox[idx_panel][0][run], recall_spox[idx_panel][0][run], f1_spox[idx_panel][0][run], y_pred_lda_test, y_pred_lda_train = LDA(
                        train_sets,
                        df_spox,
                        seeds[run],
                        run,
                        "lda_threshold",
                        feature_folder,
                        LDA_folder,
                        metrics_folder,
                        cm_folder,
                        mis_folder,
                        class_threshold_folder,
                        classified_folder,
                        unknown_pred_folder,
                        df_name_with_panel,
                        min(2, len(df_train["panel_detail"].unique()) - 1),
                        0.5,
                        True,
                        norm=True
                    )
                # X_test = X_test.iloc[:, :-1]
                # train_sets = (X_train, y_train, X_test, y_test) 

                if len(y_train.unique()) > 2:
                    accuracy[idx_panel][1][run], precision[idx_panel][1][run], recall[idx_panel][1][run], f1[idx_panel][1][run], accuracy_spox[idx_panel][1][run], precision_spox[idx_panel][1][run], recall_spox[idx_panel][1][run], f1_spox[idx_panel][1][run], y_pred_lda_test, y_pred_lda_train = LDA(
                        train_sets,
                        df_spox,
                        seeds[run],
                        run,
                        "lda",
                        feature_folder,
                        LDA_folder,
                        metrics_folder,
                        cm_folder,
                        mis_folder,
                        class_threshold_folder,
                        classified_folder,
                        unknown_pred_folder,
                        df_name_with_panel,
                        min(2, len(df_train["panel_detail"].unique()) - 1),
                        0.5,
                        False,
                        norm=True
                    )
                accuracy[idx_panel][2][run], precision[idx_panel][2][run], recall[idx_panel][2][run], f1[idx_panel][2][run], accuracy_spox[idx_panel][2][run], precision_spox[idx_panel][2][run], recall_spox[idx_panel][2][run], f1_spox[idx_panel][2][run], y_pred_rf_test, y_pred_rf_train = RF(
                    1000,
                    5,
                    train_sets,
                    df_spox,
                    seeds[run],
                    run,
                    "rf",
                    feature_folder,
                    metrics_folder,
                    cm_folder,
                    mis_folder,
                    classified_folder,
                    unknown_pred_folder,
                    df_name_with_panel,
                    None,
                    False,
                    norm=True
                )
                accuracy[idx_panel][3][run], precision[idx_panel][3][run], recall[idx_panel][3][run], f1[idx_panel][3][run], accuracy_spox[idx_panel][3][run], precision_spox[idx_panel][3][run], recall_spox[idx_panel][3][run], f1_spox[idx_panel][3][run], y_pred_rf_test, y_pred_rf_train = XGBoost(
                    1000,
                    5,
                    train_sets,
                    df_spox,
                    seeds[run],
                    run,
                    "xgboost",
                    feature_folder,
                    metrics_folder,
                    cm_folder,
                    mis_folder,
                    classified_folder,
                    unknown_pred_folder,
                    df_name_with_panel,
                    None,
                    False,
                    norm=True
                )
                if len(y_train.unique()) > 2:
                    accuracy[idx_panel][4][run], precision[idx_panel][4][run], recall[idx_panel][4][run], f1[idx_panel][4][run], accuracy_spox[idx_panel][4][run], precision_spox[idx_panel][4][run], recall_spox[idx_panel][4][run], f1_spox[idx_panel][4][run] = LDA_RF(
                        1000,
                        5,
                        train_sets,
                        df_spox,
                        seeds[run],
                        run,
                        "lda_rf",
                        metrics_folder,
                        cm_folder,
                        mis_folder,
                        classified_folder,
                        unknown_pred_folder,
                        df_name_with_panel,
                        min(2, len(df_train["panel_detail"].unique()) - 1),
                        None,
                        False,
                        norm=True
                    )
                if preprocessed_input == False:
                    accuracy[idx_panel][5][run], precision[idx_panel][5][run], recall[idx_panel][5][run], f1[idx_panel][5][run], accuracy_spox[idx_panel][5][run], precision_spox[idx_panel][5][run], recall_spox[idx_panel][5][run], f1_spox[idx_panel][5][run] = FRBC(
                        str(df_name_with_panel)+"-TRAIN",
                        str(df_name_with_panel)+"-TEST",
                        str(df_name_with_panel)+"-SPOX",
                        frbc_params,
                        5,
                        train_sets_frbc,
                        spox_set_frbc,
                        seeds[run],
                        run,
                        "frbc_threshold",
                        metrics_folder,
                        cm_folder,
                        mis_folder,
                        class_threshold_folder,
                        classified_folder,
                        rule_folder,
                        unknown_pred_folder,
                        df_name_with_panel,
                        0.5,
                        True
                    )
                    accuracy[idx_panel][6][run], precision[idx_panel][6][run], recall[idx_panel][6][run], f1[idx_panel][6][run], accuracy_spox[idx_panel][6][run], precision_spox[idx_panel][6][run], recall_spox[idx_panel][6][run], f1_spox[idx_panel][6][run] = FRBC(
                        str(df_name_with_panel)+"-TRAIN",
                        str(df_name_with_panel)+"-TEST",
                        str(df_name_with_panel)+"-SPOX",
                        frbc_params,
                        5,
                        train_sets_frbc,
                        spox_set_frbc,
                        seeds[run],
                        run,
                        "frbc",
                        metrics_folder,
                        cm_folder,
                        mis_folder,
                        class_threshold_folder,
                        classified_folder,
                        rule_folder,
                        unknown_pred_folder,
                        df_name_with_panel,
                        None,
                        False
                    )
                    if len(y_train.unique()) > 2:
                        accuracy[idx_panel][7][run], precision[idx_panel][7][run], recall[idx_panel][7][run], f1[idx_panel][7][run], accuracy_spox[idx_panel][7][run], precision_spox[idx_panel][7][run], recall_spox[idx_panel][7][run], f1_spox[idx_panel][7][run] = LDA_FRBC(
                            str(df_name_with_panel)+"-TRAIN",
                            str(df_name_with_panel)+"-TEST",
                            str(df_name_with_panel)+"-SPOX",
                            lda_frbc_params,
                            5,
                            train_sets_frbc,
                            spox_set_frbc,
                            seeds[run],
                            run,
                            "lda_frbc",
                            metrics_folder,
                            cm_folder,
                            mis_folder,
                            classified_folder,
                            rule_folder,
                            unknown_pred_folder,
                            min(2, len(df_train["panel_detail"].unique()) - 1),
                            df_name_with_panel,
                        )
                df_spox = preprocess_spox(
                    test_file,
                    filter,
                    antibody=antibody,
                    sero_th=sero_status,
                    data_column=data_col,
                    exclude_features=exclude_cols,
                    preprocessed=preprocessed_input
                )
                accuracy[idx_panel][8][run], precision[idx_panel][8][run], recall[idx_panel][8][run], f1[idx_panel][8][run], accuracy_spox[idx_panel][8][run], precision_spox[idx_panel][8][run], recall_spox[idx_panel][8][run], f1_spox[idx_panel][8][run], y_pred_rf_test, y_pred_rf_train = deeptables(
                    1000,
                    5,
                    train_sets,
                    df_spox,
                    seeds[run],
                    run,
                    "deeptables",
                    feature_folder,
                    metrics_folder,
                    cm_folder,
                    mis_folder,
                    classified_folder,
                    unknown_pred_folder,
                    df_name_with_panel,
                    None,
                    False,
                    norm=True
                )         
                              
        panel_l = [p for p in itertools.product([df_all, df_acute, df_epi], repeat=2)]
        for panel_idx, (df_train, df_test) in enumerate(panel_l):
            df_name_with_panel = f"train_{df_train.name}_test_{df_test.name}_{df_name}"
            save_statistical_report(
                accuracy[panel_idx], precision[panel_idx], recall[panel_idx], f1[panel_idx], df_name_with_panel, stat_folder
            )
            save_statistical_report(
                accuracy_spox[panel_idx], precision_spox[panel_idx], recall_spox[panel_idx], f1_spox[panel_idx], df_name_with_panel, stat_folder_spox
            )          
        

if __name__ == "__main__":
    main()
