from __future__ import print_function

import os
import gc
import os.path
import warnings

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils import *

from platypus.algorithms import *

from deeptables.models import deeptable

from sklearn.inspection import permutation_importance

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)

from skmoefs.rcs import RCSInitializer, RCSVariator
from skmoefs.discretization.discretizer_base import fuzzyDiscretization
from skmoefs.toolbox import (
    MPAES_RCS,
    load_dataset,
    normalize,
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
    print('classifiers spox_sets\n', spox_sets)
    X_spox = spox_sets.iloc[:, 1:].copy()
    y_spox = spox_sets.iloc[:, 0].copy()
    print('classifiers lda xspox\n', X_spox)

    if norm == True:
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train[X_train.columns] = min_max_scaler.fit_transform(X_train)
        X_test[X_test.columns] = min_max_scaler.transform(X_test)
        X_spox[X_spox.columns] = min_max_scaler.transform(X_spox)

    lda = LinearDiscriminantAnalysis(n_components=n_att).fit(X_train, y_train)

    y_test_pred = lda.predict(X_test)
    y_train_pred = lda.predict(X_train)
    y_spox_pred = lda.predict(X_spox)
    
    save_lda_plot(X_train, y_train, X_test, y_test_pred, lda, LDA_folder, output_name, run)

    #spox_out = X_spox.copy() # use copy so it does not change original df
    #spox_out['panel_detail'] = y_spox_prediction
    #spox_out['conf'] = conf_degrees_spox
    #spox_out = spox_out[spox_out["conf"] > threshold_value]

    conf_degrees_test = np.max(lda.predict_proba(X_test), axis = 1)
    conf_degrees_train = np.max(lda.predict_proba(X_train), axis = 1)
    conf_degrees_spox = np.max(lda.predict_proba(X_spox), axis = 1)
    
    conf_y_real, conf_y_pred = threshold_usage(lda, X_test, y_test, y_test_pred, conf_degrees_test, threshold_value)
    conf_y_real_spox, conf_y_pred_spox = threshold_usage(lda, X_spox, y_spox, y_spox_pred, conf_degrees_spox, threshold_value)
    
    if threshold_use == True and len(conf_y_real) > 0 and len(conf_y_pred) > 0:
        test = X_test
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
            spox = X_spox
            spox["conf_degree"] = conf_degrees_spox
            new_spox = spox[spox["conf_degree"] >= threshold_value]
            
            directory = str(classifier) + '_revised_data'
            parent_dir = os.path.join(class_threshold_folder, directory)
            if os.path.exists(parent_dir + "/") == False:
                os.makedirs(parent_dir)
            
            path = (
                parent_dir
                + "/"
                + str(output_name)
                + "_"
                + str(run)
                + ".csv"
            )
            
            df = new_spox.copy(deep=True)
            df["real"] = conf_y_real_spox
            df["pred"] = conf_y_pred_spox
            
            df_out = df[(df["conf_degree"] >= threshold_value) & (df["real"] == df["pred"])]
            df_out.to_csv(path)
        
            directory = str(classifier) + '_revised_data'
            parent_dir = os.path.join(mis_folder, directory)
            if os.path.exists(parent_dir + "/") == False:
                os.makedirs(parent_dir)
            
            path = (
                parent_dir
                + "/"
                + str(output_name)
                + "_"
                + str(run)
                + ".csv"
            )
            
            del df, df_out
            gc.collect()
        
            df = new_spox.copy(deep=True)
    
            df["real"] = conf_y_real_spox
            df["pred"] = conf_y_pred_spox
        
            df_out = df[df["real"] != df["pred"]]
            df_out.to_csv(path)
            
            del df, df_out
            gc.collect()
            
            accuracy_spox = accuracy_score(conf_y_real_spox, conf_y_pred_spox)
            precision_spox = precision_score(conf_y_real_spox, conf_y_pred_spox, average="macro")
            recall_spox = recall_score(conf_y_real_spox, conf_y_pred_spox, average="macro")
            f1_spox = f1_score(conf_y_real_spox, conf_y_pred_spox, average="macro")
            
            directory = str(classifier) + '_revised_data'
            parent_dir = os.path.join(metrics_folder, directory)
            if os.path.exists(parent_dir + "/") == False:
                os.makedirs(parent_dir)
            
            path = (
                parent_dir
                + "/"
                + str(output_name)
                + "_"
                + str(run)
                + ".txt"
            )
        
            f = open(path, "w")
            f.write(classification_report(conf_y_real_spox, conf_y_pred_spox))
            f.write("\n")
        
            f.write("accuracy = ")
            f.write(str(accuracy_spox))
            f.write("\n")
        
            f.write("precision = ")
            f.write(str(precision_spox))
            f.write("\n")
        
            f.write("recall = ")
            f.write(str(recall_spox))
            f.write("\n")
        
            f.write("f1 = ")
            f.write(str(f1_spox))
            f.write("\n")
            f.close()
            
            directory = str(classifier) + '_revised_data'
            parent_dir = os.path.join(cm_folder, directory)
            if os.path.exists(parent_dir + "/") == False:
                os.makedirs(parent_dir)
            
            path = (
                parent_dir
                + "/"
                + str(output_name)
                + "_"
                + str(run)
                + ".png"
            )
            
            cm = confusion_matrix(conf_y_real_spox, conf_y_pred_spox)
            cm = confusion_matrix(conf_y_real_spox, conf_y_pred_spox, labels=lda.classes_)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lda.classes_)
            disp = disp.plot(
                include_values=True, cmap="viridis", ax=None, xticks_rotation="horizontal"
            )
        
            plt.grid(False)
            plt.savefig(path)
            plt.close()
        
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
    
    else:   
        save_misclassified_data(
            X_test, y_test, y_test_pred, output_name, str(classifier), mis_folder, run, threshold_value, threshold_use
        )
        
        directory = str(classifier) + '_revised_data'
        parent_dir = os.path.join(mis_folder, directory)
        if os.path.exists(parent_dir + "/") == False:
            os.makedirs(parent_dir)
        
        path = (
            parent_dir
            + "/"
            + str(output_name)
            + "_"
            + str(run)
            + ".csv"
        )
        
        df = X_spox.copy(deep=True)

        df["real"] = y_spox
        df["pred"] = y_spox_pred
    
        df_out = df[df["real"] != df["pred"]]
        df_out.to_csv(path)
        
        del df, df_out
        gc.collect()
        
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
        
        directory = str(classifier) + '_revised_data'
        parent_dir = os.path.join(metrics_folder, directory)
        if os.path.exists(parent_dir + "/") == False:
            os.makedirs(parent_dir)
        
        path = (
            parent_dir
            + "/"
            + str(output_name)
            + "_"
            + str(run)
            + ".txt"
        )
    
        f = open(path, "w")
        f.write(classification_report(y_spox, y_spox_pred))
        f.write("\n")
    
        f.write("accuracy = ")
        f.write(str(accuracy_spox))
        f.write("\n")
    
        f.write("precision = ")
        f.write(str(precision_spox))
        f.write("\n")
    
        f.write("recall = ")
        f.write(str(recall_spox))
        f.write("\n")
    
        f.write("f1 = ")
        f.write(str(f1_spox))
        f.write("\n")
        f.close()
        
        save_confusion_matrix(y_test, y_test_pred, output_name, str(classifier), cm_folder, run, classifier=lda)
        
        directory = str(classifier) + '_revised_data'
        parent_dir = os.path.join(cm_folder, directory)
        if os.path.exists(parent_dir + "/") == False:
            os.makedirs(parent_dir)
        
        path = (
            parent_dir
            + "/"
            + str(output_name)
            + "_"
            + str(run)
            + ".png"
        )
        
        cm = confusion_matrix(y_spox, y_spox_pred)
        cm = confusion_matrix(y_spox, y_spox_pred, labels=lda.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lda.classes_)
        disp = disp.plot(
            include_values=True, cmap="viridis", ax=None, xticks_rotation="horizontal"
        )
    
        plt.grid(False)
        plt.savefig(path)
        plt.close()
            
    save_classified_general(
            X_train, X_test, y_train, y_train_pred, y_test, y_test_pred, output_name, str(classifier), class_folder, run, conf_degrees_train, conf_degrees_test, threshold_use
        )
        
    directory = str(classifier) + '_revised_data'
    parent_dir = os.path.join(class_folder, directory)
    if os.path.exists(parent_dir + "/") == False:
        os.makedirs(parent_dir)
    
    path = (
        parent_dir
        + "/"
        + str(output_name)
        + "_"
        + str(run)
        + ".csv"
    )
    
    df = X_spox.copy(deep=True)
    df["real"] = y_spox
    df["pred"] = y_spox_pred
    df.to_csv(path)
    
    del df
    gc.collect()
        
    #save_unknown_preds(spox_out, unknown_pred_folder, str(classifier), run, output_name)
    
    del X_spox, y_spox, X_test, y_test, X_train, y_train
    gc.collect()

    return accuracy, precision, recall, f1, accuracy_spox, precision_spox, recall_spox, f1_spox, y_test_pred, y_train_pred


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

    if norm == True:
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train[X_train.columns] = min_max_scaler.fit_transform(X_train)
        X_test[X_test.columns] = min_max_scaler.transform(X_test)
        X_spox[X_spox.columns] = min_max_scaler.transform(X_spox)
   

    rf = RandomForestClassifier(
        n_estimators=n_est, max_depth=depth, random_state=seed
    ).fit(X_train, y_train)

    y_test_pred = rf.predict(X_test)
    y_train_pred = rf.predict(X_train)
    y_spox_pred = rf.predict(X_spox)

    # export important features
    export_feature_importance(X_train, y_train, X_test, y_test_pred, rf, feature_folder, output_name, run)

    # prediction of unknown samples
    #y_spox_prediction = rf.predict(X_spox)
    #spox_out = X_spox.copy()
    #spox_out['panel_detail'] = y_spox_prediction

    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average="macro")
    recall = recall_score(y_test, y_test_pred, average="macro")
    f1 = f1_score(y_test, y_test_pred, average="macro")
    
    accuracy_spox = accuracy_score(y_spox, y_spox_pred)
    precision_spox = precision_score(y_spox, y_spox_pred, average="macro")
    recall_spox = recall_score(y_spox, y_spox_pred, average="macro")
    f1_spox = f1_score(y_spox, y_spox_pred, average="macro")

    sorted_idx = rf.feature_importances_.argsort()

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
    
    directory = str(classifier) + '_revised_data'
    parent_dir = os.path.join(metrics_folder, directory)
    if os.path.exists(parent_dir + "/") == False:
        os.makedirs(parent_dir)
    
    path = (
        parent_dir
        + "/"
        + str(output_name)
        + "_"
        + str(run)
        + ".txt"
    )

    f = open(path, "w")
    f.write(classification_report(y_spox, y_spox_pred))
    f.write("\n")

    f.write("accuracy = ")
    f.write(str(accuracy_spox))
    f.write("\n")

    f.write("precision = ")
    f.write(str(precision_spox))
    f.write("\n")

    f.write("recall = ")
    f.write(str(recall_spox))
    f.write("\n")

    f.write("f1 = ")
    f.write(str(f1_spox))
    f.write("\n")
    f.close()

    save_confusion_matrix(y_test, y_test_pred, output_name, str(classifier), cm_folder, run, classifier=rf)
    
    directory = str(classifier) + '_revised_data'
    parent_dir = os.path.join(cm_folder, directory)
    if os.path.exists(parent_dir + "/") == False:
        os.makedirs(parent_dir)
    
    path = (
        parent_dir
        + "/"
        + str(output_name)
        + "_"
        + str(run)
        + ".png"
    )
    
    cm = confusion_matrix(y_spox, y_spox_pred)
    cm = confusion_matrix(y_spox, y_spox_pred, labels=rf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
    disp = disp.plot(
        include_values=True, cmap="viridis", ax=None, xticks_rotation="horizontal"
    )

    plt.grid(False)
    plt.savefig(path)
    plt.close()

    save_misclassified_data(
        X_test, y_test, y_test_pred, output_name, str(classifier), mis_folder, run, threshold_value, threshold_use
    )
    
    directory = str(classifier) + '_revised_data'
    parent_dir = os.path.join(mis_folder, directory)
    if os.path.exists(parent_dir + "/") == False:
        os.makedirs(parent_dir)
    
    path = (
        parent_dir
        + "/"
        + str(output_name)
        + "_"
        + str(run)
        + ".csv"
    )
    
    df = X_spox.copy(deep=True)

    df["real"] = y_spox
    df["pred"] = y_spox_pred

    df_out = df[df["real"] != df["pred"]]
    df_out.to_csv(path)
    
    del df, df_out
    gc.collect()
    
    save_classified_general(
            X_train, X_test, y_train, y_train_pred, y_test, y_test_pred, output_name, str(classifier), class_folder, run, None, None, False
        )
        
    directory = str(classifier) + '_revised_data'
    parent_dir = os.path.join(class_folder, directory)
    if os.path.exists(parent_dir + "/") == False:
        os.makedirs(parent_dir)
    
    path = (
        parent_dir
        + "/"
        + str(output_name)
        + "_"
        + str(run)
        + ".csv"
    )
    
    df = X_spox.copy(deep=True)
    df["real"] = y_spox
    df["pred"] = y_spox_pred
    df.to_csv(path)
    
    del df
    gc.collect()
    
    #save_unknown_preds(spox_out, unknown_pred_folder, str(classifier), run, output_name)

    return accuracy, precision, recall, f1, accuracy_spox, precision_spox, recall_spox, f1_spox, y_test_pred, y_train_pred


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
    norm=True
):
    X_train, y_train, X_test, y_test = train_sets
    X_train = X_train.copy()
    y_train = y_train.copy()
    X_test = X_test.copy()
    y_test = y_test.copy()
    X_spox = spox_sets.iloc[:, 1:].copy()
    y_spox = spox_sets.iloc[:, 0].copy()

    if norm == True:
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train[X_train.columns] = min_max_scaler.fit_transform(X_train)
        X_test[X_test.columns] = min_max_scaler.transform(X_test)
        X_spox[X_spox.columns] = min_max_scaler.transform(X_spox)

    lda = LinearDiscriminantAnalysis(n_components=n_att)

    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    X_spox_lda = lda.transform(X_spox)

    rf = RandomForestClassifier(
        n_estimators=n_est, max_depth=depth, random_state=seed
    ).fit(X_train_lda, y_train)

    y_test_pred = rf.predict(X_test_lda)
    y_train_pred = rf.predict(X_train_lda)
    y_spox_pred = rf.predict(X_spox_lda)
        
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average="macro")
    recall = recall_score(y_test, y_test_pred, average="macro")
    f1 = f1_score(y_test, y_test_pred, average="macro")
    
    accuracy_spox = accuracy_score(y_spox, y_spox_pred)
    precision_spox = precision_score(y_spox, y_spox_pred, average="macro")
    recall_spox = recall_score(y_spox, y_spox_pred, average="macro")
    f1_spox = f1_score(y_spox, y_spox_pred, average="macro")

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
    
    directory = str(classifier) + '_revised_data'
    parent_dir = os.path.join(metrics_folder, directory)
    if os.path.exists(parent_dir + "/") == False:
        os.makedirs(parent_dir)
    
    path = (
        parent_dir
        + "/"
        + str(output_name)
        + "_"
        + str(run)
        + ".txt"
    )

    f = open(path, "w")
    f.write(classification_report(y_spox, y_spox_pred))
    f.write("\n")

    f.write("accuracy = ")
    f.write(str(accuracy_spox))
    f.write("\n")

    f.write("precision = ")
    f.write(str(precision_spox))
    f.write("\n")

    f.write("recall = ")
    f.write(str(recall_spox))
    f.write("\n")

    f.write("f1 = ")
    f.write(str(f1_spox))
    f.write("\n")
    f.close()

    save_confusion_matrix(y_test, y_test_pred, output_name, str(classifier), cm_folder, run, classifier=rf)
    
    directory = str(classifier) + '_revised_data'
    parent_dir = os.path.join(cm_folder, directory)
    if os.path.exists(parent_dir + "/") == False:
        os.makedirs(parent_dir)
    
    path = (
        parent_dir
        + "/"
        + str(output_name)
        + "_"
        + str(run)
        + ".png"
    )
    
    cm = confusion_matrix(y_spox, y_spox_pred)
    cm = confusion_matrix(y_spox, y_spox_pred, labels=rf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
    disp = disp.plot(
        include_values=True, cmap="viridis", ax=None, xticks_rotation="horizontal"
    )

    plt.grid(False)
    plt.savefig(path)
    plt.close()

    save_misclassified_data(
        X_test, y_test, y_test_pred, output_name, str(classifier), mis_folder, run, threshold_value, threshold_use
    )
    
    directory = str(classifier) + '_revised_data'
    parent_dir = os.path.join(mis_folder, directory)
    if os.path.exists(parent_dir + "/") == False:
        os.makedirs(parent_dir)
    
    path = (
        parent_dir
        + "/"
        + str(output_name)
        + "_"
        + str(run)
        + ".csv"
    )
    
    df = X_spox.copy(deep=True)

    df["real"] = y_spox
    df["pred"] = y_spox_pred

    df_out = df[df["real"] != df["pred"]]
    df_out.to_csv(path)
    
    del df, df_out
    gc.collect()
    
    save_classified_general(
            X_train, X_test, y_train, y_train_pred, y_test, y_test_pred, output_name, str(classifier), class_folder, run, None, None, False
        )
        
    directory = str(classifier) + '_revised_data'
    parent_dir = os.path.join(class_folder, directory)
    if os.path.exists(parent_dir + "/") == False:
        os.makedirs(parent_dir)
    
    path = (
        parent_dir
        + "/"
        + str(output_name)
        + "_"
        + str(run)
        + ".csv"
    )
    
    df = X_spox.copy(deep=True)
    df["real"] = y_spox
    df["pred"] = y_spox_pred
    df.to_csv(path)
    
    del df
    gc.collect()

    #save_unknown_preds(spox_out, unknown_pred_folder, str(classifier), run, output_name)

    return accuracy, precision, recall, f1, accuracy_spox, precision_spox, recall_spox, f1_spox


def FRBC(
    dataset_train,
    dataset_test,
    dataset_spox, #added to classify unlabelled data
    params,
    fuzzy_sets,
    train_sets,
    spox_set, #added to classify unlabelled data
    seed,
    run,
    classifier_name,
    metrics_folder,
    cm_folder,
    mis_folder,
    class_threshold_folder,
    class_folder,
    rule_folder,
    unknown_pred_folder, #folder for spox output
    output_name,
    threshold_value,
    threshold_use,
):
    M = params[0]
    Amin = params[1]
    nEvals = params[2]
    capacity = params[3]
    divisions = params[4]
    
    algs = ["moead", "nsga3", "spea2", "mpaes22"]

    variator = RCSVariator()
    discretizer = fuzzyDiscretization(numSet=fuzzy_sets)
    initializer = RCSInitializer(discretizer=discretizer)

    set_rng(seed)

    X_train, y_train, attributes_train, inputs, outputs = load_dataset(dataset_train, False)
    X_test, y_test, attributes_test, inputs, outputs = load_dataset(dataset_test, False)
    X_spox, y_spox, attributes_spox, inputs, outputs = load_dataset(dataset_spox, False)

    X_train, y_train = normalize(X_train, y_train, attributes_train)
    X_test, y_test = normalize(X_test, y_test, attributes_test)
    X_spox, y_spox = normalize(X_spox, y_spox, attributes_spox)

    tmp_acc = []
    classifiers = []
    for alg in algs:
        mpaes_rcs_fdt = MPAES_RCS(
            M=M,
            Amin=Amin,
            capacity=capacity,
            divisions=divisions,
            variator=variator,
            initializer=initializer,
            moea_type=alg,
            objectives=["accuracy", "trl"],
        )
    
        mpaes_rcs_fdt.fit(X_train, y_train, max_evals=nEvals)

        index = mpaes_rcs_fdt._from_position_to_index("first")
        if index is not None:
            classifiers.append(mpaes_rcs_fdt.classifiers[index])
            
        y_pred = classifiers[-1].predict(X_train)
        tmp_acc.append(accuracy_score(y_train, y_pred))
        
    best_idx = np.argmax(np.array(tmp_acc))
    classifier = classifiers[best_idx]

    y_spox_pred = classifier.predict(X_spox) 
    y_test_pred = classifier.predict(X_test)
    y_train_pred = classifier.predict(X_train)
    
    conf_degrees_spox = classifier.conf_degree(X_spox)
    conf_degrees_test = classifier.conf_degree(X_test)
    conf_degrees_train = classifier.conf_degree(X_train)
    if threshold_use == True:
        conf_y_real, conf_y_pred = threshold_usage(classifier, X_test, y_test, y_test_pred, conf_degrees_test, threshold_value)
        conf_y_real_spox, conf_y_pred_spox = threshold_usage(classifier, X_spox, y_spox, y_spox_pred, conf_degrees_spox, threshold_value)
    
    spox = spox_set
    train, test = train_sets
    
    replace_panel(test)
    replace_panel(spox)
    
    if threshold_use == True and len(conf_y_real) > 0 and len(conf_y_pred) > 0:
        test["conf_degree"] = conf_degrees_test
        
        new_test = test[test["conf_degree"] >= threshold_value]
        
        save_classified_with_threshold(
            new_test, conf_y_real, conf_y_pred, output_name, str(classifier_name), class_threshold_folder, run, threshold_value, threshold_use
        )
        
        save_misclassified_data(
            new_test, conf_y_real, conf_y_pred, output_name, str(classifier_name), mis_folder, run, threshold_value, threshold_use
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
            str(classifier_name),
            metrics_folder,
            run,
        )
    
        save_confusion_matrix(conf_y_real, conf_y_pred, output_name, str(classifier_name), cm_folder, run, classifier=None)
        
        if len(conf_y_real_spox) > 0 and len(conf_y_pred_spox) > 0:
            spox["conf_degree"] = conf_degrees_spox
            
            new_spox = spox[spox["conf_degree"] >= threshold_value]
            
            directory = str(classifier_name) + '_revised_data'
            parent_dir = os.path.join(class_threshold_folder, directory)
            if os.path.exists(parent_dir + "/") == False:
                os.makedirs(parent_dir)
            
            path = (
                parent_dir
                + "/"
                + str(output_name)
                + "_"
                + str(run)
                + ".csv"
            )
            
            df = new_spox.copy(deep=True)
            df["real"] = conf_y_real_spox
            df["pred"] = conf_y_pred_spox
            
            df_out = df[(df["conf_degree"] >= threshold_value) & (df["real"] == df["pred"])]
            df_out.to_csv(path)
            
            del df, df_out
            gc.collect()
            
            accuracy_spox = accuracy_score(conf_y_real_spox, conf_y_pred_spox)
            precision_spox = precision_score(conf_y_real_spox, conf_y_pred_spox, average="macro")
            recall_spox = recall_score(conf_y_real_spox, conf_y_pred_spox, average="macro")
            f1_spox = f1_score(conf_y_real_spox, conf_y_pred_spox, average="macro")
            
            directory = str(classifier_name) + '_revised_data'
            parent_dir = os.path.join(metrics_folder, directory)
            if os.path.exists(parent_dir + "/") == False:
                os.makedirs(parent_dir)
            
            path = (
                parent_dir
                + "/"
                + str(output_name)
                + "_"
                + str(run)
                + ".txt"
            )
        
            f = open(path, "w")
            f.write(classification_report(conf_y_real_spox, conf_y_pred_spox))
            f.write("\n")
        
            f.write("accuracy = ")
            f.write(str(accuracy_spox))
            f.write("\n")
        
            f.write("precision = ")
            f.write(str(precision_spox))
            f.write("\n")
        
            f.write("recall = ")
            f.write(str(recall_spox))
            f.write("\n")
        
            f.write("f1 = ")
            f.write(str(f1_spox))
            f.write("\n")
            f.close()
            
            directory = str(classifier_name) + '_revised_data'
            parent_dir = os.path.join(cm_folder, directory)
            if os.path.exists(parent_dir + "/") == False:
                os.makedirs(parent_dir)
            
            path = (
                parent_dir
                + "/"
                + str(output_name)
                + "_"
                + str(run)
                + ".png"
            )
            
            cm = confusion_matrix(conf_y_real_spox, conf_y_pred_spox)
            cm = confusion_matrix(conf_y_real_spox, conf_y_pred_spox)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp = disp.plot(
                include_values=True, cmap="viridis", ax=None, xticks_rotation="horizontal"
            )
        
            plt.grid(False)
            plt.savefig(path)
            plt.close()
            
        save_misclassified_data(
            new_test, conf_y_real, conf_y_pred, output_name, str(classifier_name), mis_folder, run, threshold_value, threshold_use
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
            str(classifier_name),
            metrics_folder,
            run,
        )
    
        save_confusion_matrix(conf_y_real, conf_y_pred, output_name, str(classifier_name), cm_folder, run, classifier=None)
            
    else:    
        save_misclassified_data(
            test.iloc[:, :-1], test.iloc[:, -1], y_test_pred, output_name, str(classifier_name), mis_folder, run, threshold_value, threshold_use
        )
        
        directory = str(classifier_name) + '_revised_data'
        parent_dir = os.path.join(mis_folder, directory)
        if os.path.exists(parent_dir + "/") == False:
            os.makedirs(parent_dir)
        
        path = (
            parent_dir
            + "/"
            + str(output_name)
            + "_"
            + str(run)
            + ".csv"
        )
        
        df = spox.iloc[:, :-1].copy(deep=True)

        df["real"] = spox.iloc[:, -1]
        df["pred"] = y_spox_pred
    
        df_out = df[df["real"] != df["pred"]]
        df_out.to_csv(path)
        
        del df, df_out
        gc.collect()
        
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
            str(classifier_name),
            metrics_folder,
            run,
        )
        
        accuracy_spox = accuracy_score(y_spox, y_spox_pred)
        precision_spox = precision_score(y_spox, y_spox_pred, average="macro")
        recall_spox = recall_score(y_spox, y_spox_pred, average="macro")
        f1_spox = f1_score(y_spox, y_spox_pred, average="macro")
        
        directory = str(classifier_name) + '_revised_data'
        parent_dir = os.path.join(metrics_folder, directory)
        if os.path.exists(parent_dir + "/") == False:
            os.makedirs(parent_dir)
        
        path = (
            parent_dir
            + "/"
            + str(output_name)
            + "_"
            + str(run)
            + ".txt"
        )
    
        f = open(path, "w")
        f.write(classification_report(y_spox, y_spox_pred))
        f.write("\n")
    
        f.write("accuracy = ")
        f.write(str(accuracy_spox))
        f.write("\n")
    
        f.write("precision = ")
        f.write(str(precision_spox))
        f.write("\n")
    
        f.write("recall = ")
        f.write(str(recall_spox))
        f.write("\n")
    
        f.write("f1 = ")
        f.write(str(f1_spox))
        f.write("\n")
        f.close()
    
        save_confusion_matrix(y_test, y_test_pred, output_name, str(classifier_name), cm_folder, run, classifier=None)
        
        directory = str(classifier_name) + '_revised_data'
        parent_dir = os.path.join(cm_folder, directory)
        if os.path.exists(parent_dir + "/") == False:
            os.makedirs(parent_dir)
        
        path = (
            parent_dir
            + "/"
            + str(output_name)
            + "_"
            + str(run)
            + ".png"
        )
        
        cm = confusion_matrix(y_spox, y_spox_pred)
        cm = confusion_matrix(y_spox, y_spox_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp = disp.plot(
            include_values=True, cmap="viridis", ax=None, xticks_rotation="horizontal"
        )
    
        plt.grid(False)
        plt.savefig(path)
        plt.close()
        
    save_classified_general(
        train.iloc[:, :-1], test.iloc[:, :-1], train.iloc[:, -1], y_train_pred, test.iloc[:, -1], y_test_pred, output_name, str(classifier_name), class_folder, run, conf_degrees_train, conf_degrees_test, threshold_use
    )
    
    directory = str(classifier_name) + '_revised_data'
    parent_dir = os.path.join(class_folder, directory)
    if os.path.exists(parent_dir + "/") == False:
        os.makedirs(parent_dir)
    
    path = (
        parent_dir
        + "/"
        + str(output_name)
        + "_"
        + str(run)
        + ".csv"
    )
    
    df = spox.iloc[:, :-1].copy(deep=True)
    df["real"] = spox.iloc[:, -1]
    df["pred"] = y_spox_pred
    df.to_csv(path)
    
    del df
    gc.collect()
        
    #save_unknown_preds(spox_out, unknown_pred_folder, str(classifier_name), run, output_name) #save results for spox data

    directory = str(classifier_name)
    new_dir = os.path.join(rule_folder, directory)
    
    if os.path.exists(new_dir + "/") == False:
        os.mkdir(new_dir)
    path = new_dir + "/" + str(output_name) + "_" + str(run) + ".txt"

    file_ = open(path, "a")
    classifier.show_RB(inputs, outputs, f=file_)
    file_.close()

    return accuracy, precision, recall, f1, accuracy_spox, precision_spox, recall_spox, f1_spox


def LDA_FRBC(
    dataset_train,
    dataset_test,
    dataset_spox, #added to classify unlabelled data
    params,
    fuzzy_sets,
    train_sets,
    spox_set, #added to classify unlabelled data
    seed,
    run,
    classifier_name,
    metrics_folder,
    cm_folder,
    mis_folder,
    class_folder,
    rule_folder,
    unknown_pred_folder, #folder for spox output
    n_att,
    output_name
):
    M = params[0]
    Amin = params[1]
    nEvals = params[2]
    capacity = params[3]
    divisions = params[4]
    
    algs = ["moead", "nsga3", "spea2", "mpaes22"]

    variator = RCSVariator()
    discretizer = fuzzyDiscretization(numSet=fuzzy_sets)
    initializer = RCSInitializer(discretizer=discretizer)

    set_rng(seed)
    
    X_train, y_train, attributes_train, inputs, outputs = load_dataset(dataset_train, False)
    X_test, y_test, attributes_test, inputs, outputs = load_dataset(dataset_test, False)
    X_spox, y_spox, attributes_spox, inputs, outputs = load_dataset(dataset_spox, False)

    X_train, y_train = normalize(X_train, y_train, attributes_train)
    X_test, y_test = normalize(X_test, y_test, attributes_test)
    X_spox, y_spox = normalize(X_spox, y_spox, attributes_spox)

    lda = LinearDiscriminantAnalysis(n_components=n_att)

    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    X_spox_lda = lda.transform(X_spox)
    
    tmp_acc = []
    classifiers = []
    for alg in algs:
        mpaes_rcs_fdt = MPAES_RCS(
            M=M,
            Amin=Amin,
            capacity=capacity,
            divisions=divisions,
            variator=variator,
            initializer=initializer,
            moea_type=alg,
            objectives=["accuracy", "trl"],
        )
    
        mpaes_rcs_fdt.fit(X_train_lda, y_train, max_evals=nEvals)

        index = mpaes_rcs_fdt._from_position_to_index("first")
        if index is not None:
            classifiers.append(mpaes_rcs_fdt.classifiers[index])
            
        y_pred = classifiers[-1].predict(X_train_lda)
        tmp_acc.append(accuracy_score(y_train, y_pred))
        
    best_idx = np.argmax(np.array(tmp_acc))
    classifier = classifiers[best_idx]

    y_spox_pred = classifier.predict(X_spox_lda)
    y_test_pred = classifier.predict(X_test_lda)
    y_train_pred = classifier.predict(X_train_lda)
    
    train, test = train_sets
    spox = spox_set
    replace_panel(test)
    
    save_misclassified_data(
            test.iloc[:, :-1], test.iloc[:, -1], y_test_pred, output_name, str(classifier_name), mis_folder, run, None, False
        )
        
    directory = str(classifier_name) + '_revised_data'
    parent_dir = os.path.join(mis_folder, directory)
    if os.path.exists(parent_dir + "/") == False:
        os.makedirs(parent_dir)
    
    path = (
        parent_dir
        + "/"
        + str(output_name)
        + "_"
        + str(run)
        + ".csv"
    )
    
    df = spox.iloc[:, :-1].copy(deep=True)

    df["real"] = spox.iloc[:, -1]
    df["pred"] = y_spox_pred

    df_out = df[df["real"] != df["pred"]]
    df_out.to_csv(path)
    
    del df, df_out
    gc.collect()
        
    save_classified_general(
            train.iloc[:, :-1], test.iloc[:, :-1], train.iloc[:, -1], y_train_pred, test.iloc[:, -1], y_test_pred, output_name, str(classifier_name), class_folder, run, None, None, False
        )
        
    directory = str(classifier_name) + '_revised_data'
    parent_dir = os.path.join(class_folder, directory)
    if os.path.exists(parent_dir + "/") == False:
        os.makedirs(parent_dir)
    
    path = (
        parent_dir
        + "/"
        + str(output_name)
        + "_"
        + str(run)
        + ".csv"
    )
    
    df = spox.iloc[:, :-1].copy(deep=True)
    df["real"] = spox.iloc[:, -1]
    df["pred"] = y_spox_pred
    df.to_csv(path)
    
    del df
    gc.collect()

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
        str(classifier_name),
        metrics_folder,
        run,
    )
    
    accuracy_spox = accuracy_score(y_spox, y_spox_pred)
    precision_spox = precision_score(y_spox, y_spox_pred, average="macro")
    recall_spox = recall_score(y_spox, y_spox_pred, average="macro")
    f1_spox = f1_score(y_spox, y_spox_pred, average="macro")
    
    directory = str(classifier_name) + '_revised_data'
    parent_dir = os.path.join(metrics_folder, directory)
    if os.path.exists(parent_dir + "/") == False:
        os.makedirs(parent_dir)
    
    path = (
        parent_dir
        + "/"
        + str(output_name)
        + "_"
        + str(run)
        + ".txt"
    )

    f = open(path, "w")
    f.write(classification_report(y_spox, y_spox_pred))
    f.write("\n")

    f.write("accuracy = ")
    f.write(str(accuracy_spox))
    f.write("\n")

    f.write("precision = ")
    f.write(str(precision_spox))
    f.write("\n")

    f.write("recall = ")
    f.write(str(recall_spox))
    f.write("\n")

    f.write("f1 = ")
    f.write(str(f1_spox))
    f.write("\n")
    f.close()

    save_confusion_matrix(y_test, y_test_pred, output_name, str(classifier_name), cm_folder, run, classifier=None)
    
    directory = str(classifier_name) + '_revised_data'
    parent_dir = os.path.join(cm_folder, directory)
    if os.path.exists(parent_dir + "/") == False:
        os.makedirs(parent_dir)
    
    path = (
        parent_dir
        + "/"
        + str(output_name)
        + "_"
        + str(run)
        + ".png"
    )
    
    cm = confusion_matrix(y_spox, y_spox_pred)
    cm = confusion_matrix(y_spox, y_spox_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp = disp.plot(
        include_values=True, cmap="viridis", ax=None, xticks_rotation="horizontal"
    )

    plt.grid(False)
    plt.savefig(path)
    plt.close()

    parent_dir = rule_folder
    directory = str(classifier_name)
    new_dir = os.path.join(parent_dir, directory)

    if os.path.exists(new_dir + "/") == False:
        os.mkdir(new_dir)
    path = new_dir + "/" + str(output_name) + "_" + str(run) + ".txt"

    file_ = open(path, "a")
    classifier.show_RB(inputs, outputs, f=file_)
    file_.close()

    return accuracy, precision, recall, f1, accuracy_spox, precision_spox, recall_spox, f1_spox

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
    norm=True
):

    X_train, y_train, X_test, y_test = train_sets
    X_train = X_train.copy()
    y_train = y_train.copy()
    X_test = X_test.copy()
    y_test = y_test.copy()
    X_spox = spox_sets.iloc[:, 1:].copy()
    y_spox = spox_sets.iloc[:, 0].copy()

    if norm == True:
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train[X_train.columns] = min_max_scaler.fit_transform(X_train)
        X_test[X_test.columns] = min_max_scaler.transform(X_test)
        X_spox[X_spox.columns] = min_max_scaler.transform(X_spox)

    rf = GradientBoostingClassifier(
        n_estimators=n_est, max_depth=depth, random_state=seed
    ).fit(X_train, y_train)

    y_test_pred = rf.predict(X_test)
    y_spox_pred = rf.predict(X_spox)
    y_train_pred = rf.predict(X_train)

    export_feature_importance(X_train, y_train, X_test, y_test_pred, rf, feature_folder, output_name, run)

    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average="macro")
    recall = recall_score(y_test, y_test_pred, average="macro")
    f1 = f1_score(y_test, y_test_pred, average="macro")
    
    accuracy_spox = accuracy_score(y_spox, y_spox_pred)
    precision_spox = precision_score(y_spox, y_spox_pred, average="macro")
    recall_spox = recall_score(y_spox, y_spox_pred, average="macro")
    f1_spox = f1_score(y_spox, y_spox_pred, average="macro")

    sorted_idx = rf.feature_importances_.argsort()

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
    
    directory = str(classifier) + '_revised_data'
    parent_dir = os.path.join(metrics_folder, directory)
    if os.path.exists(parent_dir + "/") == False:
        os.makedirs(parent_dir)
    
    path = (
        parent_dir
        + "/"
        + str(output_name)
        + "_"
        + str(run)
        + ".txt"
    )

    f = open(path, "w")
    f.write(classification_report(y_spox, y_spox_pred))
    f.write("\n")

    f.write("accuracy = ")
    f.write(str(accuracy_spox))
    f.write("\n")

    f.write("precision = ")
    f.write(str(precision_spox))
    f.write("\n")

    f.write("recall = ")
    f.write(str(recall_spox))
    f.write("\n")

    f.write("f1 = ")
    f.write(str(f1_spox))
    f.write("\n")
    f.close()

    save_confusion_matrix(y_test, y_test_pred, output_name, str(classifier), cm_folder, run, classifier=rf)
    
    directory = str(classifier) + '_revised_data'
    parent_dir = os.path.join(cm_folder, directory)
    if os.path.exists(parent_dir + "/") == False:
        os.makedirs(parent_dir)
    
    path = (
        parent_dir
        + "/"
        + str(output_name)
        + "_"
        + str(run)
        + ".png"
    )
    
    cm = confusion_matrix(y_spox, y_spox_pred)
    cm = confusion_matrix(y_spox, y_spox_pred, labels=rf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
    disp = disp.plot(
        include_values=True, cmap="viridis", ax=None, xticks_rotation="horizontal"
    )
    
    plt.grid(False)
    plt.savefig(path)
    plt.close()

    save_misclassified_data(
        X_test, y_test, y_test_pred, output_name, str(classifier), mis_folder, run, threshold_value, threshold_use
    )
    
    directory = str(classifier) + '_revised_data'
    parent_dir = os.path.join(mis_folder, directory)
    if os.path.exists(parent_dir + "/") == False:
        os.makedirs(parent_dir)
    
    path = (
        parent_dir
        + "/"
        + str(output_name)
        + "_"
        + str(run)
        + ".csv"
    )
    
    df = X_spox.copy(deep=True)

    df["real"] = y_spox
    df["pred"] = y_spox_pred

    df_out = df[df["real"] != df["pred"]]
    df_out.to_csv(path)
    
    del df, df_out
    gc.collect()
    
    save_classified_general(
            X_train, X_test, y_train, y_train_pred, y_test, y_test_pred, output_name, str(classifier), class_folder, run, None, None, False
        )
        
    directory = str(classifier) + '_revised_data'
    parent_dir = os.path.join(class_folder, directory)
    if os.path.exists(parent_dir + "/") == False:
        os.makedirs(parent_dir)
    
    path = (
        parent_dir
        + "/"
        + str(output_name)
        + "_"
        + str(run)
        + ".csv"
    )
    
    df = X_spox.copy(deep=True)
    df["real"] = y_spox
    df["pred"] = y_spox_pred
    df.to_csv(path)
    
    del df
    gc.collect()
    
    #save_unknown_preds(spox_out, unknown_pred_folder, str(classifier), run, output_name)

    return accuracy, precision, recall, f1, accuracy_spox, precision_spox, recall_spox, f1_spox, y_test_pred, y_train_pred
    
    
def deeptables(
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
    
    print('deeptables\n', X_spox)

    if norm == True:
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train[X_train.columns] = min_max_scaler.fit_transform(X_train)
        X_test[X_test.columns] = min_max_scaler.transform(X_test)
        X_spox[X_spox.columns] = min_max_scaler.transform(X_spox)

    conf = deeptable.ModelConfig(nets=['dnn_nets'], optimizer=tf.keras.optimizers.RMSprop(), earlystopping_patience=10)

    dt = deeptable.DeepTable(config=conf)

    model, history = dt.fit(X_train, y_train, epochs=10)

    y_test_pred = dt.predict(X_test)
    y_train_pred = dt.predict(X_train)
    y_spox_pred = dt.predict(X_spox)

    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average="macro")
    recall = recall_score(y_test, y_test_pred, average="macro")
    f1 = f1_score(y_test, y_test_pred, average="macro")
    
    accuracy_spox = accuracy_score(y_spox, y_spox_pred)
    precision_spox = precision_score(y_spox, y_spox_pred, average="macro")
    recall_spox = recall_score(y_spox, y_spox_pred, average="macro")
    f1_spox = f1_score(y_spox, y_spox_pred, average="macro")
    
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
    
    directory = str(classifier) + '_revised_data'
    parent_dir = os.path.join(metrics_folder, directory)
    if os.path.exists(parent_dir + "/") == False:
        os.makedirs(parent_dir)
    
    path = (
        parent_dir
        + "/"
        + str(output_name)
        + "_"
        + str(run)
        + ".txt"
    )

    f = open(path, "w")
    f.write(classification_report(y_spox, y_spox_pred))
    f.write("\n")

    f.write("accuracy = ")
    f.write(str(accuracy_spox))
    f.write("\n")

    f.write("precision = ")
    f.write(str(precision_spox))
    f.write("\n")

    f.write("recall = ")
    f.write(str(recall_spox))
    f.write("\n")

    f.write("f1 = ")
    f.write(str(f1_spox))
    f.write("\n")
    f.close()

    save_confusion_matrix(y_test, y_test_pred, output_name, str(classifier), cm_folder, run, classifier=dt)

    directory = str(classifier) + '_revised_data'
    parent_dir = os.path.join(cm_folder, directory)
    if os.path.exists(parent_dir + "/") == False:
        os.makedirs(parent_dir)
    
    path = (
        parent_dir
        + "/"
        + str(output_name)
        + "_"
        + str(run)
        + ".png"
    )
    
    cm = confusion_matrix(y_spox, y_spox_pred)
    cm = confusion_matrix(y_spox, y_spox_pred, labels=dt.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt.classes_)
    disp = disp.plot(
        include_values=True, cmap="viridis", ax=None, xticks_rotation="horizontal"
    )

    plt.grid(False)
    plt.savefig(path)
    plt.close()
    
    save_misclassified_data(
        X_test, y_test, y_test_pred, output_name, str(classifier), mis_folder, run, threshold_value, threshold_use
    )
    
    directory = str(classifier) + '_revised_data'
    parent_dir = os.path.join(mis_folder, directory)
    if os.path.exists(parent_dir + "/") == False:
        os.makedirs(parent_dir)
    
    path = (
        parent_dir
        + "/"
        + str(output_name)
        + "_"
        + str(run)
        + ".csv"
    )
    
    df = X_spox.copy(deep=True)

    df["real"] = y_spox
    df["pred"] = y_spox_pred

    df_out = df[df["real"] != df["pred"]]
    df_out.to_csv(path)
    
    del df, df_out
    gc.collect()
    
    save_classified_general(
            X_train, X_test, y_train, y_train_pred, y_test, y_test_pred, output_name, str(classifier), class_folder, run, None, None, False
        )
        
    directory = str(classifier) + '_revised_data'
    parent_dir = os.path.join(class_folder, directory)
    if os.path.exists(parent_dir + "/") == False:
        os.makedirs(parent_dir)
    
    path = (
        parent_dir
        + "/"
        + str(output_name)
        + "_"
        + str(run)
        + ".csv"
    )
    
    df = X_spox.copy(deep=True)
    df["real"] = y_spox
    df["pred"] = y_spox_pred
    df.to_csv(path)
    
    del df
    gc.collect()
    
    #save_unknown_preds(spox_out, unknown_pred_folder, str(classifier), run, output_name)

    return accuracy, precision, recall, f1, accuracy_spox, precision_spox, recall_spox, f1_spox, y_test_pred, y_train_pred