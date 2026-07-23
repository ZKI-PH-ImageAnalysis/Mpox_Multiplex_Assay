import click
import pathlib

import numpy as np
import pandas as pd

from classifiers import *

import os
import math
import itertools
from sklearn.model_selection import StratifiedKFold

antibody_list = ["IgM_IgG", "IgG"]
sero_list = ["all"]
data_cols_list = [ "dataIn"]
exclusion_list = [("L1R", "M1", "VACV")] 


def group_df_analyte(df, data_column="data"):
    df = df.pivot_table(
        values=data_column,
        index=["sampleID_metadata", "panel_detail", "panel"],
        columns=["analyte"],
        aggfunc="first",
        dropna=True,
    )
    df = df.dropna()
    df = df.reset_index(level=["panel_detail"])
    return df


def preprocess_spox(
    input_file,
    filter_csv,
    antibody="IgG",
    sero_th="all",
    data_column="dataIn",
    exclude_features=("None"),
    preprocessed=False,
):
    df_in = pd.read_csv(input_file, low_memory=False)

    is_instrument_file = "sample_category" in df_in.columns or "highBG" in df_in.columns

    if "sample_name" in df_in.columns:
        df_in = df_in.rename(columns={"sample_name": "sampleID_meta"})

    if "sampleID_meta" not in df_in.columns:
        raise ValueError("No sample ID column found (expected sampleID_meta or sample_name)")

    if is_instrument_file:
        # Map instrument labels -> model classes
        mapping = {
            "Pos": "MPXV",
            "Pos.": "MPXV",
            "Pos_Vax": "MPXV",
            "Vax": "MVA",
            "Vax.": "MVA",
            "Neg": "Pre",
            "Neg.": "Pre",
        }

        if "panel" not in df_in.columns:
            df_in["panel"] = "Pre"
            print("No panel column found, defaulting to 'Pre' for all samples.")
        else:
            df_in["panel"] = df_in["sample_category"].map(mapping)

        if df_in["panel"].isna().any():
            unknown = df_in[df_in["panel"].isna()]["sample_category"].unique()
            raise ValueError(f"Unknown sample_category values: {unknown}")

        # No serostatus filtering for instrument data
        serostatus_IDs = None

    else:
        # Original dataset behavior
        df_in["serostatus_delta_IgG"] = df_in.get("serostatus_cat.delta", np.nan)

        serostatus_IDs = df_in[df_in["serostatus_delta_IgG"].notna()]

        if sero_th == "positive":
            serostatus_IDs = serostatus_IDs[serostatus_IDs["serostatus_delta_IgG"].isin(["positive"])]
        elif sero_th == "borderline positive":
            serostatus_IDs = serostatus_IDs[
                serostatus_IDs["serostatus_delta_IgG"].isin(["borderline positive", "positive"])
            ]

        serostatus_IDs = serostatus_IDs["sampleID_meta"].unique()


    cols_to_drop = df_in.columns[df_in.columns.str.endswith(exclude_features)]
    df_in = df_in.drop(cols_to_drop, axis=1, errors="ignore")

    df_in.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(
        f"ATTENTION: Dataframe includes {df_in.isna().sum().sum()} NaNs. "
        f"These will be excluded."
    )
    # Replace inf with NaN first
    df_in.replace([np.inf, -np.inf], np.nan, inplace=True)

    total_nans = df_in.isna().sum().sum()
    print(f"ATTENTION: Dataframe includes {total_nans} NaNs.")

    # Find rows containing at least one NaN
    rows_with_nan = df_in[df_in.isna().any(axis=1)]

    if len(rows_with_nan) > 0:
        print(f"\nRemoving {len(rows_with_nan)} rows because they contain NaN values:\n")

        for idx, row in rows_with_nan.iterrows():
            nan_cols = row[row.isna()].index.tolist()
            sample = row.get("sampleID_meta", idx)

            print(
                f"Row index={idx}, sampleID={sample}, "
                f"NaN columns={nan_cols}"
            )
    else:
        print("No rows contain NaN values.")

    df_in = df_in.dropna()

    dataIn_columns = [col for col in df_in.columns if col.startswith("dataIn")]
    keep_cols = ["sampleID_meta", "panel"]

    def transform_row(row):
        out = {}
        isotype = row.get("isotype", "IgG")
        for col in dataIn_columns:
            base = col.split("_", 1)[1]
            out[f"{isotype}_{base}"] = row[col]
        return pd.Series(out)

    transformed = df_in.apply(transform_row, axis=1)
    df_in = pd.concat([df_in[keep_cols], transformed], axis=1)

    # collapse duplicates per sample
    df_in = df_in.groupby("sampleID_meta", as_index=False).first()

    if antibody != "IgM_IgG":
        antibody_cols = [c for c in df_in.columns if c.startswith(antibody)]
        df_in = df_in[["sampleID_meta", "panel"] + antibody_cols]

    df_in["panel_detail"] = df_in["panel"]

    # filter serostatus if applicable (only old dataset)
    if serostatus_IDs is not None:
        df_in = df_in[df_in["sampleID_meta"].isin(serostatus_IDs)]

    df_in = df_in[df_in["panel"] != "CPXV"]
    df_in = df_in[df_in["panel"] != "SPox"]
    df_in = df_in[df_in["panel"] != "SPox_Rep"]

    # unify naming
    df_in.loc[df_in.panel == "Pre_New", "panel"] = "Pre"
    df_in.loc[df_in.panel_detail == "Pre_New", "panel_detail"] = "Pre"

    df_spox = df_in.drop(["panel"], axis=1)
    df_spox = df_spox.set_index("sampleID_meta")

    # sort columns (important for ML consistency)
    cols = ["panel_detail"] + sorted([c for c in df_spox.columns if c != "panel_detail"])
    df_spox = df_spox[cols]

    if filter_csv is not None and os.path.isfile(filter_csv):
        df_filter = pd.read_csv(filter_csv)
        df_filter = df_filter.rename(columns={"excludeIDs": "sampleID_meta"})

        df_spox = df_spox.reset_index()

        df_spox = df_spox.merge(df_filter, on="sampleID_meta", how="left", indicator=True)
        df_spox = df_spox[df_spox["_merge"] == "left_only"].drop(columns=["_merge"])

        df_spox = df_spox.set_index("sampleID_meta")


    rows_with_nan = df_spox[df_spox.isna().any(axis=1)]
    if len(rows_with_nan) > 0:
        print(f"\nFinal dataframe: removing {len(rows_with_nan)} rows with NaN values:\n")

        for idx, row in rows_with_nan.iterrows():
            nan_cols = row[row.isna()].index.tolist()
            print(
                f"sampleID={idx}, "
                f"NaN columns={nan_cols}"
            )

    df_spox = df_spox.dropna()

    return df_spox


def preprocess_data(
    df_in,
    test_file,
    filter_csv,
    antibody="IgG",
    sero_th="all",
    data_column="data",
    exclude_features=("None"),
    preprocessed=False,
    antigen_to_remove = ""
):
    """
    Antibody: "IgG", "IgM", "IgM_IgG"
    sero_th: "all", "positive", "borderline positive"
    data_column: "data", "dataln"
    panel: "all", "acute", "epi"
    exclude_features: list of features we would like to exclude for example ["M1", "L1R"]
    """
    df_out = None
    if df_in is not None:
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
        # Remove CPXV for now
        df_out = df_out[df_out["panel_detail"] != "CPXV"]
        df_rep = df_out[df_out["panel_detail"] == "SPox_Rep"].drop(["panel"], axis=1)
        # Add -rep to ID
        df_rep = df_rep.rename(index=lambda s: s + '-rep')
    
    

    # Extract the unknown samples as df_spox
    #df_spox = df_out[df_out["panel_detail"] == "SPox"].drop(["panel"], axis=1)

    # Join the filtered out samples to the Spox dataframe
    #if filter_csv is not None and os.path.isfile(filter_csv):
    #    df_joined = df_joined.drop("panel", axis=1)
    #    df_joined = df_joined.set_index(["sampleID_metadata"])
    #    df_spox = pd.concat([df_spox, df_joined])


    # extract the repetition panel
    
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
    

    df_all, df_acute, df_epi = None, None, None
    if df_out is not None:
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

        if antigen_to_remove:
            if isinstance(antigen_to_remove, str):
                antigens = [antigen_to_remove]
            else:
                antigens = list(antigen_to_remove)

            cols_to_remove = [f"{iso}_{ag}" for iso in ("IgG", "IgM") for ag in antigens]
            for df in [df_all, df_acute, df_epi, df_spox]:
                df.drop(columns=[col for col in cols_to_remove if col in df.columns], inplace=True)
    

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

def store_run_result(target, idx_panel, alg_idx, run, result):
    (
        target["accuracy"][idx_panel][alg_idx][run],
        target["precision"][idx_panel][alg_idx][run],
        target["recall"][idx_panel][alg_idx][run],
        target["f1"][idx_panel][alg_idx][run],
        target["accuracy_spox"][idx_panel][alg_idx][run],
        target["precision_spox"][idx_panel][alg_idx][run],
        target["recall_spox"][idx_panel][alg_idx][run],
        target["f1_spox"][idx_panel][alg_idx][run],
        _,
        _,
    ) = result


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["train", "inference"]),
    default="train",
    help="Run mode: 'train' (default) or 'inference' (load model and predict).",
)
@click.option(
    "--model-path",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path),
    default=None,
    help="Path to saved model (used in inference mode)",
)
@click.option(
    "--n-splits",
    type=int,
    default=5,
    help="Number of folds for StratifiedKFold (overrides default)",
)
@click.option(
    "--reps",
    type=int,
    default=3,
    help="Number of repetitions for CV (overrides default)",
)
@click.option(
    "--start-seed",
    type=int,
    default=70,
    help="Start seed for repeated CV (overrides default)",
)
@click.option(
    "--input-file",
    type=click.Path(
        file_okay=True, dir_okay=False, path_type=pathlib.Path
    ),
    help = "Path to dataInput.csv",
    default = None,
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
@click.option(
    "--antigen-to-remove",
    type=str,
    multiple=True,
    help = "Bool value if inputfile csv is already preprocessed",
    default = [],
)
def main(mode, model_path, n_splits, reps, start_seed, input_file, test_file, filter, outdir, preprocessed_input, antigen_to_remove):
    df_assay = None
    if input_file:         
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
            antigen_to_remove=antigen_to_remove,
        )
        # just for readability when saving files
        exclude_cols = "".join(exclude_cols)
        d[
            f"antibody_{antibody}_serostatus_{sero_status}_datacol_{data_col}_excluding_{exclude_cols}"
        ] = [df_all, df_acute, df_epi, df_spox]

    # If inference mode, do a simple prediction pass using provided model
    if mode == "inference":
        if model_path is None:
            raise click.BadParameter("--model-path is required in inference mode")

        from utils import load_model
        model = load_model(str(model_path))

        for df_name, [df_all, df_acute, df_epi, df_spox] in d.items():
            print(f"Running inference on dataset variant: {df_name}")
            X_spox = df_spox.iloc[:, 1:]
            min_max_scaler = preprocessing.MinMaxScaler()
            # Attention. Here we are applying the same scaling as in training, but we should ideally save the scaler from training and load it here to ensure consistency. 
            # For now we are just fitting a new scaler on the spox data, which is not ideal but will have to do for this demonstration.
            X_spox[X_spox.columns] = min_max_scaler.fit_transform(X_spox)

            preds = model.predict(X_spox)

            outdir_pred = os.path.join(outdir, "inference_preds")
            os.makedirs(outdir_pred, exist_ok=True)
            path = os.path.join(outdir_pred, f"preds_{df_name}.csv")
            out_df = df_spox.copy(deep=True)
            out_df["pred"] = preds
            out_df.to_csv(path)
            print(f"Saved predictions to {path}")

        return

    # override CV settings from CLI args
    n_split = n_splits
    seed_runs = reps * n_split
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

    active_algorithms = [
        {
            "slot": 3,
            "runner": XGBoost,
            "name": "xgboost",
            "n_est": 1000,
            "depth": 5,
        },
        #{
        #    "slot": 8,
        #    "runner": TabPFN,
        #    "name": "tabpfn",
        #    "n_est": 1000,
        #    "depth": 5,
        #},
    ]

    # for dataframe in dataframe_list
    for df_name, [df_all, df_acute, df_epi, df_spox] in d.items():
        df_all.name = "all"
        df_acute.name = "acute"
        df_epi.name = "epi"
        df_spox.name = "spox"
        panel_l = [p for p in itertools.product([df_all, df_acute, df_epi], repeat=2)]

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
        
                df_name_with_panel = (
                    f"train_{df_train.columns.name }_test_{df_test.columns.name }_{df_name}"
                )
                train_sets = (X_train, y_train, X_test, y_test)

                run_targets = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "accuracy_spox": accuracy_spox,
                    "precision_spox": precision_spox,
                    "recall_spox": recall_spox,
                    "f1_spox": f1_spox,
                }

                for spec in active_algorithms:
                    result = spec["runner"](
                        spec["n_est"],
                        spec["depth"],
                        train_sets,
                        df_spox,
                        seeds[run],
                        run,
                        spec["name"],
                        feature_folder,
                        metrics_folder,
                        cm_folder,
                        mis_folder,
                        classified_folder,
                        unknown_pred_folder,
                        df_name_with_panel,
                        None,
                        False,
                        norm=True,
                    )
                    store_run_result(run_targets, idx_panel, spec["slot"], run, result)
                              
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
