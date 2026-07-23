from __future__ import print_function

import os
import gc
import warnings

import numpy as np
import matplotlib.pyplot as plt

from utils import *

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

warnings.filterwarnings("ignore")


def threshold_usage(classifier, X_test, y_test, y_test_pred, conf_degrees, threshold_value):
    conf_y_real = []
    conf_y_pred = []

    for i in range(y_test.shape[0]):
        if conf_degrees[i] > threshold_value:
            conf_y_real.append(y_test[i])
            conf_y_pred.append(y_test_pred[i])

    return conf_y_real, conf_y_pred


def replace_panel(df):
    df.replace({"panel_detail": "MPXV"}, {"panel_detail": 0}, inplace=True)
    df.replace({"panel_detail": "MVA"}, {"panel_detail": 1}, inplace=True)
    df.replace({"panel_detail": "Pre"}, {"panel_detail": 2}, inplace=True)


def _split_inputs(train_sets, spox_sets, norm=True):
    X_train, y_train, X_test, y_test = train_sets
    X_train = X_train.copy()
    y_train = y_train.copy()
    X_test = X_test.copy()
    y_test = y_test.copy()
    X_spox = spox_sets.iloc[:, 1:].copy()
    y_spox = spox_sets.iloc[:, 0].copy()

    if norm:
        scaler = preprocessing.MinMaxScaler()
        X_train[X_train.columns] = scaler.fit_transform(X_train)
        X_test[X_test.columns] = scaler.transform(X_test)
        X_spox[X_spox.columns] = scaler.transform(X_spox)

    return X_train, y_train, X_test, y_test, X_spox, y_spox


def _score_predictions(y_true, y_pred):
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, average="macro"),
        recall_score(y_true, y_pred, average="macro"),
        f1_score(y_true, y_pred, average="macro"),
    )


def _save_report(path, y_true, y_pred):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(classification_report(y_true, y_pred))
        f.write("\n")


def _save_confusion_plot(folder, output_name, classifier_name, run, y_true, y_pred, labels=None):
    parent_dir = os.path.join(folder, str(classifier_name) + "_revised_data")
    os.makedirs(parent_dir, exist_ok=True)
    path = os.path.join(parent_dir, f"{output_name}_{run}.png")

    cm = confusion_matrix(y_true, y_pred, labels=labels) if labels is not None else confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels) if labels is not None else ConfusionMatrixDisplay(confusion_matrix=cm)
    disp = disp.plot(include_values=True, cmap="viridis", ax=None, xticks_rotation="horizontal")
    plt.grid(False)
    plt.savefig(path)
    plt.close()


def _save_prediction_csv(folder, output_name, classifier_name, run, X_data, y_real, y_pred):
    parent_dir = os.path.join(folder, str(classifier_name) + "_revised_data")
    os.makedirs(parent_dir, exist_ok=True)
    path = os.path.join(parent_dir, f"{output_name}_{run}.csv")
    df = X_data.copy(deep=True)
    df["real"] = y_real
    df["pred"] = y_pred
    df.to_csv(path)
    return path


def _save_common_outputs(
    model,
    classifier_name,
    output_name,
    run,
    X_train,
    X_test,
    y_train,
    y_train_pred,
    y_test,
    y_test_pred,
    X_spox,
    y_spox,
    y_spox_pred,
    metrics_folder,
    cm_folder,
    mis_folder,
    class_folder,
    threshold_use,
    threshold_value,
    model_name,
    conf_degrees_train=None,
    conf_degrees_test=None,
    save_model_name=True,
):
    test_metrics = _score_predictions(y_test, y_test_pred)
    spox_metrics = _score_predictions(y_spox, y_spox_pred)

    save_metrics(y_test, y_test_pred, *test_metrics, output_name, str(classifier_name), metrics_folder, run)
    save_confusion_matrix(y_test, y_test_pred, output_name, str(classifier_name), cm_folder, run, classifier=model)
    save_misclassified_data(
        X_test,
        y_test,
        y_test_pred,
        output_name,
        str(classifier_name),
        mis_folder,
        run,
        threshold_value,
        threshold_use,
    )
    _save_report(
        os.path.join(metrics_folder, str(classifier_name) + "_revised_data", f"{output_name}_{run}.txt"),
        y_spox,
        y_spox_pred,
    )
    _save_confusion_plot(cm_folder, output_name, classifier_name, run, y_spox, y_spox_pred, labels=getattr(model, "classes_", None))
    save_classified_general(
        X_train,
        X_test,
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
        output_name,
        str(classifier_name),
        class_folder,
        run,
        conf_degrees_train,
        conf_degrees_test,
        threshold_use,
    )
    _save_prediction_csv(class_folder, output_name, classifier_name, run, X_spox, y_spox, y_spox_pred)
    if save_model_name:
        try:
            save_model(model, os.path.join(metrics_folder, "models"), model_name)
        except Exception:
            pass

    return test_metrics + spox_metrics


def LDA(
    train_sets,
    spox_sets,
    seed,
    run,
    classifier,
    feature_folder,
    LDA_folder,
    metrics_folder,
    cm_folder,
    mis_folder,
    class_threshold_folder,
    class_folder,
    unknown_pred_folder,
    output_name,
    n_att,
    threshold_value,
    threshold_use,
    norm=True,
):
    X_train, y_train, X_test, y_test = train_sets
    X_train = X_train.copy()
    y_train = y_train.copy()
    X_test = X_test.copy()
    y_test = y_test.copy()

    X_spox = spox_sets.iloc[:, 1:].copy()
    y_spox = spox_sets.iloc[:, 0].copy()

    if norm:
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train[X_train.columns] = min_max_scaler.fit_transform(X_train)
        X_test[X_test.columns] = min_max_scaler.transform(X_test)
        X_spox[X_spox.columns] = min_max_scaler.transform(X_spox)

    lda = LinearDiscriminantAnalysis(n_components=n_att).fit(X_train, y_train)

    y_test_pred = lda.predict(X_test)
    y_train_pred = lda.predict(X_train)
    y_spox_pred = lda.predict(X_spox)

    save_lda_plot(X_train, y_train, X_test, y_test_pred, lda, LDA_folder, output_name, run)

    conf_degrees_test = np.max(lda.predict_proba(X_test), axis=1)
    conf_degrees_train = np.max(lda.predict_proba(X_train), axis=1)
    conf_degrees_spox = np.max(lda.predict_proba(X_spox), axis=1)

    conf_y_real, conf_y_pred = threshold_usage(lda, X_test, y_test, y_test_pred, conf_degrees_test, threshold_value)
    conf_y_real_spox, conf_y_pred_spox = threshold_usage(lda, X_spox, y_spox, y_spox_pred, conf_degrees_spox, threshold_value)

    if threshold_use and len(conf_y_real) > 0 and len(conf_y_pred) > 0:
        test = X_test.copy()
        test["conf_degree"] = conf_degrees_test
        new_test = test[test["conf_degree"] >= threshold_value]

        save_classified_with_threshold(
            new_test, conf_y_real, conf_y_pred, output_name, str(classifier), class_threshold_folder, run, threshold_value, threshold_use
        )

        save_misclassified_data(
            new_test, conf_y_real, conf_y_pred, output_name, str(classifier), mis_folder, run, threshold_value, threshold_use
        )

        accuracy = accuracy_score(conf_y_real, conf_y_pred)
        precision = precision_score(conf_y_real, conf_y_pred, average="macro")
        recall = recall_score(conf_y_real, conf_y_pred, average="macro")
        f1 = f1_score(conf_y_real, conf_y_pred, average="macro")

        save_metrics(
            conf_y_real,
            conf_y_pred,
            accuracy,
            precision,
            recall,
            f1,
            output_name,
            str(classifier),
            metrics_folder,
            run,
        )

        save_confusion_matrix(conf_y_real, conf_y_pred, output_name, str(classifier), cm_folder, run, classifier=lda)

        if len(conf_y_real_spox) > 0 and len(conf_y_pred_spox) > 0:
            spox = X_spox.copy()
            spox["conf_degree"] = conf_degrees_spox
            new_spox = spox[spox["conf_degree"] >= threshold_value]

            directory = str(classifier) + "_revised_data"
            parent_dir = os.path.join(class_threshold_folder, directory)
            os.makedirs(parent_dir, exist_ok=True)

            path = os.path.join(parent_dir, f"{output_name}_{run}.csv")

            df = new_spox.copy(deep=True)
            df["real"] = conf_y_real_spox
            df["pred"] = conf_y_pred_spox

            df_out = df[(df["conf_degree"] >= threshold_value) & (df["real"] == df["pred"])]
            df_out.to_csv(path)

            directory = str(classifier) + "_revised_data"
            parent_dir = os.path.join(mis_folder, directory)
            os.makedirs(parent_dir, exist_ok=True)

            path = os.path.join(parent_dir, f"{output_name}_{run}.csv")

            df = new_spox.copy(deep=True)
            df["real"] = conf_y_real_spox
            df["pred"] = conf_y_pred_spox

            df_out = df[df["real"] != df["pred"]]
            df_out.to_csv(path)

            accuracy_spox = accuracy_score(conf_y_real_spox, conf_y_pred_spox)
            precision_spox = precision_score(conf_y_real_spox, conf_y_pred_spox, average="macro")
            recall_spox = recall_score(conf_y_real_spox, conf_y_pred_spox, average="macro")
            f1_spox = f1_score(conf_y_real_spox, conf_y_pred_spox, average="macro")

            directory = str(classifier) + "_revised_data"
            parent_dir = os.path.join(metrics_folder, directory)
            os.makedirs(parent_dir, exist_ok=True)

            path = os.path.join(parent_dir, f"{output_name}_{run}.txt")

            with open(path, "w") as f:
                f.write(classification_report(conf_y_real_spox, conf_y_pred_spox))
                f.write("\n")
                f.write("accuracy = ")
                f.write(str(accuracy_spox) + "\n")
                f.write("precision = ")
                f.write(str(precision_spox) + "\n")
                f.write("recall = ")
                f.write(str(recall_spox) + "\n")
                f.write("f1 = ")
                f.write(str(f1_spox) + "\n")

            directory = str(classifier) + "_revised_data"
            parent_dir = os.path.join(cm_folder, directory)
            os.makedirs(parent_dir, exist_ok=True)

            path = os.path.join(parent_dir, f"{output_name}_{run}.png")

            cm = confusion_matrix(conf_y_real_spox, conf_y_pred_spox, labels=lda.classes_)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lda.classes_)
            disp = disp.plot(include_values=True, cmap="viridis", ax=None, xticks_rotation="horizontal")

            plt.grid(False)
            plt.savefig(path)
            plt.close()

        save_metrics(
            conf_y_real,
            conf_y_pred,
            accuracy,
            precision,
            recall,
            f1,
            output_name,
            str(classifier),
            metrics_folder,
            run,
        )

        save_confusion_matrix(conf_y_real, conf_y_pred, output_name, str(classifier), cm_folder, run, classifier=lda)

    else:
        save_misclassified_data(
            X_test, y_test, y_test_pred, output_name, str(classifier), mis_folder, run, threshold_value, threshold_use
        )

        directory = str(classifier) + "_revised_data"
        parent_dir = os.path.join(mis_folder, directory)
        os.makedirs(parent_dir, exist_ok=True)

        path = os.path.join(parent_dir, f"{output_name}_{run}.csv")

        df = X_spox.copy(deep=True)
        df["real"] = y_spox
        df["pred"] = y_spox_pred

        df_out = df[df["real"] != df["pred"]]
        df_out.to_csv(path)

        accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average="macro")
        recall = recall_score(y_test, y_test_pred, average="macro")
        f1 = f1_score(y_test, y_test_pred, average="macro")

        save_metrics(
            y_test,
            y_test_pred,
            accuracy,
            precision,
            recall,
            f1,
            output_name,
            str(classifier),
            metrics_folder,
            run,
        )

        accuracy_spox = accuracy_score(y_spox, y_spox_pred)
        precision_spox = precision_score(y_spox, y_spox_pred, average="macro")
        recall_spox = recall_score(y_spox, y_spox_pred, average="macro")
        f1_spox = f1_score(y_spox, y_spox_pred, average="macro")

        directory = str(classifier) + "_revised_data"
        parent_dir = os.path.join(metrics_folder, directory)
        os.makedirs(parent_dir, exist_ok=True)

        path = os.path.join(parent_dir, f"{output_name}_{run}.txt")

        with open(path, "w") as f:
            f.write(classification_report(y_spox, y_spox_pred))
            f.write("\n")
            f.write("accuracy = ")
            f.write(str(accuracy_spox) + "\n")
            f.write("precision = ")
            f.write(str(precision_spox) + "\n")
            f.write("recall = ")
            f.write(str(recall_spox) + "\n")
            f.write("f1 = ")
            f.write(str(f1_spox) + "\n")

        save_confusion_matrix(y_test, y_test_pred, output_name, str(classifier), cm_folder, run, classifier=lda)

        directory = str(classifier) + "_revised_data"
        parent_dir = os.path.join(cm_folder, directory)
        os.makedirs(parent_dir, exist_ok=True)

        path = os.path.join(parent_dir, f"{output_name}_{run}.png")

        cm = confusion_matrix(y_spox, y_spox_pred, labels=lda.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lda.classes_)
        disp = disp.plot(include_values=True, cmap="viridis", ax=None, xticks_rotation="horizontal")

        plt.grid(False)
        plt.savefig(path)
        plt.close()

    save_classified_general(
        X_train,
        X_test,
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
        output_name,
        str(classifier),
        class_folder,
        run,
        conf_degrees_train,
        conf_degrees_test,
        threshold_use,
    )

    directory = str(classifier) + "_revised_data"
    parent_dir = os.path.join(class_folder, directory)
    os.makedirs(parent_dir, exist_ok=True)

    path = os.path.join(parent_dir, f"{output_name}_{run}.csv")

    df = X_spox.copy(deep=True)
    df["real"] = y_spox
    df["pred"] = y_spox_pred
    df.to_csv(path)

    del df
    gc.collect()

    # save trained model
    try:
        model_dir = os.path.join(metrics_folder, "models")
        save_model(lda, model_dir, f"lda_{output_name}_{run}.joblib")
    except Exception:
        pass

    del X_spox, y_spox, X_test, y_test, X_train, y_train
    gc.collect()

    return (
        accuracy,
        precision,
        recall,
        f1,
        accuracy_spox,
        precision_spox,
        recall_spox,
        f1_spox,
        y_test_pred,
        y_train_pred,
    )


def RF(
    n_est,
    depth,
    train_sets,
    spox_sets,
    seed,
    run,
    classifier,
    feature_folder,
    metrics_folder,
    cm_folder,
    mis_folder,
    class_folder,
    unknown_pred_folder,
    output_name,
    threshold_value,
    threshold_use,
    norm=True,
):
    X_train, y_train, X_test, y_test = train_sets
    X_train = X_train.copy()
    y_train = y_train.copy()
    X_test = X_test.copy()
    y_test = y_test.copy()
    X_spox = spox_sets.iloc[:, 1:].copy()
    y_spox = spox_sets.iloc[:, 0].copy()

    if norm:
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train[X_train.columns] = min_max_scaler.fit_transform(X_train)
        X_test[X_test.columns] = min_max_scaler.transform(X_test)
        X_spox[X_spox.columns] = min_max_scaler.transform(X_spox)

    rf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=seed).fit(
        X_train, y_train
    )

    y_test_pred = rf.predict(X_test)
    y_train_pred = rf.predict(X_train)
    y_spox_pred = rf.predict(X_spox)

    export_feature_importance(X_train, y_train, X_test, y_test_pred, rf, feature_folder, output_name, run)

    result = _save_common_outputs(
        rf,
        classifier,
        output_name,
        run,
        X_train,
        X_test,
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
        X_spox,
        y_spox,
        y_spox_pred,
        metrics_folder,
        cm_folder,
        mis_folder,
        class_folder,
        threshold_use,
        threshold_value,
        f"rf_{output_name}_{run}.joblib",
    )

    return result + (y_test_pred, y_train_pred)


def LDA_RF(
    n_est,
    depth,
    train_sets,
    spox_sets,
    seed,
    run,
    classifier,
    metrics_folder,
    cm_folder,
    mis_folder,
    class_folder,
    unknown_pred_folder,
    output_name,
    n_att,
    threshold_value,
    threshold_use,
    norm=True,
):
    X_train, y_train, X_test, y_test = train_sets
    X_train = X_train.copy()
    y_train = y_train.copy()
    X_test = X_test.copy()
    y_test = y_test.copy()
    X_spox = spox_sets.iloc[:, 1:].copy()
    y_spox = spox_sets.iloc[:, 0].copy()

    if norm:
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train[X_train.columns] = min_max_scaler.fit_transform(X_train)
        X_test[X_test.columns] = min_max_scaler.transform(X_test)
        X_spox[X_spox.columns] = min_max_scaler.transform(X_spox)

    lda = LinearDiscriminantAnalysis(n_components=n_att)

    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    X_spox_lda = lda.transform(X_spox)

    rf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=seed).fit(
        X_train_lda, y_train
    )

    y_test_pred = rf.predict(X_test_lda)
    y_train_pred = rf.predict(X_train_lda)
    y_spox_pred = rf.predict(X_spox_lda)

    return _save_common_outputs(
        rf,
        classifier,
        output_name,
        run,
        X_train_lda,
        X_test_lda,
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
        X_spox_lda,
        y_spox,
        y_spox_pred,
        metrics_folder,
        cm_folder,
        mis_folder,
        class_folder,
        threshold_use,
        threshold_value,
        f"lda_rf_{output_name}_{run}.joblib",
    )


def TabPFN(
    n_est,
    depth,
    train_sets,
    spox_sets,
    seed,
    run,
    classifier,
    feature_folder,
    metrics_folder,
    cm_folder,
    mis_folder,
    class_folder,
    unknown_pred_folder,
    output_name,
    threshold_value,
    threshold_use,
    norm=True,
):
    X_train, y_train, X_test, y_test = train_sets
    X_train = X_train.copy()
    y_train = y_train.copy()
    X_test = X_test.copy()
    y_test = y_test.copy()
    X_spox = spox_sets.iloc[:, 1:].copy()
    y_spox = spox_sets.iloc[:, 0].copy()

    if norm:
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train[X_train.columns] = min_max_scaler.fit_transform(X_train)
        X_test[X_test.columns] = min_max_scaler.transform(X_test)
        X_spox[X_spox.columns] = min_max_scaler.transform(X_spox)

    from tabpfn import TabPFNClassifier
    from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
    from tabpfn.finetuning.finetuned_classifier import (
        FinetunedTabPFNClassifier,
    )

    rf = FinetunedTabPFNClassifier(device="cuda", epochs=30, learning_rate=1e-5).fit(X_train, y_train)

    y_test_pred = rf.predict(X_test)
    y_spox_pred = rf.predict(X_spox)
    y_train_pred = rf.predict(X_train)

    result = _save_common_outputs(
        rf,
        classifier,
        output_name,
        run,
        X_train,
        X_test,
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
        X_spox,
        y_spox,
        y_spox_pred,
        metrics_folder,
        cm_folder,
        mis_folder,
        class_folder,
        threshold_use,
        threshold_value,
        f"tabpfn_{output_name}_{run}.joblib",
    )

    return result + (y_test_pred, y_train_pred)


def XGBoost(
    n_est,
    depth,
    train_sets,
    spox_sets,
    seed,
    run,
    classifier,
    feature_folder,
    metrics_folder,
    cm_folder,
    mis_folder,
    class_folder,
    unknown_pred_folder,
    output_name,
    threshold_value,
    threshold_use,
    norm=True,
):
    X_train, y_train, X_test, y_test = train_sets
    X_train = X_train.copy()
    y_train = y_train.copy()
    X_test = X_test.copy()
    y_test = y_test.copy()
    X_spox = spox_sets.iloc[:, 1:].copy()
    y_spox = spox_sets.iloc[:, 0].copy()

    if norm:
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train[X_train.columns] = min_max_scaler.fit_transform(X_train)
        X_test[X_test.columns] = min_max_scaler.transform(X_test)
        missing_cols = set(X_train.columns) - set(X_spox.columns)
        extra_cols = set(X_spox.columns) - set(X_train.columns)

        # add missing columns as 0
        for c in missing_cols:
            X_spox[c] = 0
        # drop unknown columns
        X_spox = X_spox.drop(columns=list(extra_cols))
        # reorder to match training exactly
        X_spox = X_spox[X_train.columns]
        X_spox[X_spox.columns] = min_max_scaler.transform(X_spox)

    rf = GradientBoostingClassifier(n_estimators=n_est, max_depth=depth, random_state=seed).fit(
        X_train, y_train
    )

    y_test_pred = rf.predict(X_test)
    y_spox_pred = rf.predict(X_spox)
    y_train_pred = rf.predict(X_train)

    export_feature_importance(X_train, y_train, X_test, y_test_pred, rf, feature_folder, output_name, run)

    result = _save_common_outputs(
        rf,
        classifier,
        output_name,
        run,
        X_train,
        X_test,
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
        X_spox,
        y_spox,
        y_spox_pred,
        metrics_folder,
        cm_folder,
        mis_folder,
        class_folder,
        threshold_use,
        threshold_value,
        f"xgboost_{output_name}_{run}.joblib",
        save_model_name=True,
    )

    return result + (y_test_pred, y_train_pred)
