from __future__ import print_function

import math
import random
import os.path
import warnings
import statistics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.inspection import permutation_importance

from classifiers import *

from platypus.algorithms import *

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

from skmoefs.rcs import RCSInitializer, RCSVariator
from skmoefs.discretization.discretizer_base import fuzzyDiscretization
from skmoefs.toolbox import (
    MPAES_RCS,
    load_dataset,
    normalize,
    is_object_present,
    store_object,
    load_object,
)

warnings.filterwarnings("ignore")


variator = RCSVariator()
discretizer = fuzzyDiscretization(numSet=5)
initializer = RCSInitializer(discretizer=discretizer)


def set_rng(seed):
    np.random.seed(seed)
    random.seed(seed)


def frbc_get_params(alg_name):
    M = None
    Amin = None
    nEvals = None
    capacity = None
    divisions = None
    alg = None
    
    M = 200
    Amin = 1
    nEvals = 5000
    capacity = 32
    divisions = 8

    params = []
    params.append(M)
    params.append(Amin)
    params.append(nEvals)
    params.append(capacity)
    params.append(divisions)
    params.append(alg)

    return params


def lda_frbc_get_params(alg_name):
    M = None
    Amin = None
    nEvals = None
    capacity = None
    divisions = None
    alg = None

    M = 50
    Amin = 1
    nEvals = 5000
    capacity = 32
    divisions = 8

    params = []
    params.append(M)
    params.append(Amin)
    params.append(nEvals)
    params.append(capacity)
    params.append(divisions)
    params.append(alg)

    return params


def save_unknown_preds(df, folder, classifier_name, run, output_name):
    if classifier_name == "lda":
        directory = "lda"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "lda_threshold":
        directory = "lda_threshold"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "rf":
        directory = "rf"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "lda_rf":
        directory = "lda_rf"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "frbc":
        directory = "frbc"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "frbc_threshold":
        directory = "frbc_threshold"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "lda_frbc":
        directory = "lda_frbc"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "ensemble":
        directory = "ensemble"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "xgboost":
        directory = "xgboost"
        parent_dir = os.path.join(folder, directory)

    if os.path.exists(parent_dir + "/") == False:
        # os.mkdir(parent_dir)
        os.makedirs(parent_dir)
    
    path = (
        parent_dir
        + "/"
        + str(classifier_name)
        + "_"
        + str(output_name)
        + "_"
        + str(run)
        + ".csv"
    )

    df.to_csv(path)  
    
def save_metrics(
    y_test,
    y_test_pred,
    accuracy,
    precision,
    recall,
    f1,
    output_name,
    classifier_name,
    folder,
    run,
):
    parent_dir = None
    directory = None
    if classifier_name == "lda":
        directory = "lda"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "lda_threshold":
        directory = "lda_threshold"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "rf":
        directory = "rf"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "lda_rf":
        directory = "lda_rf"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "frbc":
        directory = "frbc"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "frbc_threshold":
        directory = "frbc_threshold"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "lda_frbc":
        directory = "lda_frbc"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "ensemble":
        directory = "ensemble"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "xgboost":
        directory = "xgboost"
        parent_dir = os.path.join(folder, directory)

    if os.path.exists(parent_dir + "/") == False:
        # os.mkdir(parent_dir)
        os.makedirs(parent_dir)

    path = (
        parent_dir
        + "/"
        + str(classifier_name)
        + "_"
        + str(output_name)
        + "_"
        + str(run)
        + ".txt"
    )

    f = open(path, "w")
    f.write(classification_report(y_test, y_test_pred))
    f.write("\n")

    f.write("accuracy = ")
    f.write(str(accuracy))
    f.write("\n")

    f.write("precision = ")
    f.write(str(precision))
    f.write("\n")

    f.write("recall = ")
    f.write(str(recall))
    f.write("\n")

    f.write("f1 = ")
    f.write(str(f1))
    f.write("\n")
    f.close()


def save_confusion_matrix(
    y_test, y_test_pred, output_name, classifier_name, folder, run, classifier
):
    parent_dir = None
    directory = None
    if classifier_name == "lda":
        directory = "lda"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "lda_threshold":
        directory = "lda_threshold"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "rf":
        directory = "rf"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "lda_rf":
        directory = "lda_rf"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "frbc":
        directory = "frbc"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "frbc_threshold":
        directory = "frbc_threshold"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "lda_frbc":
        directory = "lda_frbc"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "ensemble":
        directory = "ensemble"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "xgboost":
        directory = "xgboost"
        parent_dir = os.path.join(folder, directory)

    if os.path.exists(parent_dir + "/") == False:
        # os.mkdir(parent_dir)
        os.makedirs(parent_dir)

    path = (
        parent_dir
        + "/"
        + str(classifier_name)
        + "_"
        + str(output_name)
        + "_"
        + str(run)
        + ".png"
    )
    cm = confusion_matrix(y_test, y_test_pred)
    if classifier_name in ["lda", "rf", "lda_rf"]:
        cm = confusion_matrix(y_test, y_test_pred, labels=classifier.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    else:
        cm = confusion_matrix(y_test, y_test_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp = disp.plot(
        include_values=True, cmap="viridis", ax=None, xticks_rotation="horizontal"
    )

    plt.grid(False)
    plt.savefig(path)
    plt.close()


def save_misclassified_data(
    X_test, 
    y_test, 
    y_test_pred, 
    output_name, 
    classifier_name, 
    folder, 
    run, 
    threshold_value, 
    threshold_use, 
    ensemble_v="v1"
):
    parent_dir = None
    directory = None
    if classifier_name == "lda":
        directory = "lda"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "lda_threshold":
        directory = "lda_threshold"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "rf":
        directory = "rf"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "lda_rf":
        directory = "lda_rf"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "frbc":
        directory = "frbc"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "frbc_threshold":
        directory = "frbc_threshold"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "lda_frbc":
        directory = "lda_frbc"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "ensemble":
        directory = "ensemble"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "xgboost":
        directory = "xgboost"
        parent_dir = os.path.join(folder, directory)

    if os.path.exists(parent_dir + "/") == False:
        # os.mkdir(parent_dir)
        os.makedirs(parent_dir)

    path = (
        parent_dir
        + "/"
        + str(classifier_name)
        + "_"
        + str(output_name)
        + "_"
        + str(run)
        + ".csv"
    )

    if ensemble_v == "v3":
        if classifier_name != "ensemble":
            df = X_test.copy(deep=True)
        else:
            df = pd.DataFrame(X_test)
        if classifier_name == "ensemble":
            print(df)
        df["real"] = pd.Series(y_test)
        df["pred"] = pd.Series(y_test_pred)
    else:
        df = X_test.copy(deep=True)

        df["real"] = y_test
        df["pred"] = y_test_pred
    
    df_out = df[df["real"] != df["pred"]]
    df_out.to_csv(path)


def save_classified_with_threshold(
    X_test, y_test, y_test_pred, output_name, classifier_name, folder, run, threshold_value, threshold_use
):
    parent_dir = None
    directory = None
    if classifier_name == "lda_threshold":
        directory = "lda_threshold"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "frbc_threshold":
        directory = "frbc_threshold"
        parent_dir = os.path.join(folder, directory)

    if os.path.exists(parent_dir + "/") == False:
        # os.mkdir(parent_dir)
        os.makedirs(parent_dir)

    path = (
        parent_dir
        + "/"
        + str(classifier_name)
        + "_"
        + str(output_name)
        + "_"
        + str(run)
        + ".csv"
    )

    df = X_test.copy(deep=True)
    df["real"] = y_test
    df["pred"] = y_test_pred
    
    if threshold_use == True:   
        df_out = df[(df["conf_degree"] > threshold_value) & (df["real"] == df["pred"])]
        df_out.to_csv(path)
        
        
def save_classified_general(
    X_train, X_test, y_train, y_train_pred, y_test, y_test_pred, output_name, classifier_name, folder, run, conf_degrees_train, conf_degrees_test, threshold_use
):
    if not os.path.exists(folder):
        os.makedirs(folder)
    if classifier_name == "lda":
        directory = "lda"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "lda_threshold":
        directory = "lda_threshold"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "rf":
        directory = "rf"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "lda_rf":
        directory = "lda_rf"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "frbc":
        directory = "frbc"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "frbc_threshold":
        directory = "frbc_threshold"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "lda_frbc":
        directory = "lda_frbc"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "ensemble":
        directory = "ensemble"
        parent_dir = os.path.join(folder, directory)
    elif classifier_name == "xgboost":
        directory = "xgboost"
        parent_dir = os.path.join(folder, directory)

    if os.path.exists(parent_dir + "/") == False:
        # os.mkdir(parent_dir)
        os.makedirs(parent_dir)

    path_train = (
        parent_dir
        + "/"
        + str(classifier_name)
        + "_"
        + str(output_name)
        + "_train_"
        + str(run)
        + ".csv"
    )
    
    path_test = (
        parent_dir
        + "/"
        + str(classifier_name)
        + "_"
        + str(output_name)
        + "_test_"
        + str(run)
        + ".csv"
    )
    
    df_train = X_train.copy(deep=True)
    df_train["real"] = y_train
    df_train["pred"] = y_train_pred
    if threshold_use == True:
        df_train["conf_degrees"] = conf_degrees_train
    df_train.to_csv(path_train)
    
    df_test = X_test.copy(deep=True)
    df_test["real"] = y_test
    df_test["pred"] = y_test_pred
    if threshold_use == True:
        df_test["conf_degrees"] = conf_degrees_test
    df_test.to_csv(path_test)


def save_lda_plot(X_train, y_train, X_test, y_test_pred, lda, outdir, output_name, run):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    target_names = pd.unique(y_train)
    y1 = lda.transform(X_test)
    for l,c,m in zip(target_names,['r','g','b'],['s','x','o']):
        plt.scatter(y1[y_test_pred==l, 0],
                    y1[y_test_pred==l, 1],
                    c=c, marker=m, label=l,edgecolors='black')
    
    x1 = np.array([np.min(y1, axis=1), np.max(y1, axis=1)])
    for i, c in enumerate(['r','g','b']):
        b, w1, w2 = lda.intercept_[i], lda.coef_[i][0], lda.coef_[i][1]
        y2 = -(b+x1*w1)/w2    
        plt.plot(x1, y2,c=c)    
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("LDA of Test dataset")
    out_path = os.path.join(outdir, f"{output_name}_{run}")
    plt.savefig(out_path)
    plt.close()



def export_feature_importance(X_train, y_train, X_test, y_test, classifier, outdir, output_name, run):
    if not os.path.exists(outdir):
        os.makedirs(outdir)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    result = permutation_importance(classifier, X_train, y_train, n_repeats=10, random_state=42, n_jobs=2)
    perm_sorted_idx = result.importances_mean.argsort()
    ax1.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,
        labels=X_test.columns[perm_sorted_idx],
    )
    ax1.axvline(x=0, color="k", linestyle="--")
    ax1.set_xlabel("Decrease in accuracy score")

    mdi_importances = pd.Series(classifier.feature_importances_, index=X_train.columns)
    mdi_importances.sort_values().plot.barh(ax=ax2)
    ax2.set_xlabel("Gini importance")
    fig.suptitle(
    "Permutation vs. Impurity-based importances on features (train set)"
    )
    out_path = os.path.join(outdir, f"{output_name}_{run}_{classifier.__class__.__name__}_TRAIN.png")
    fig.savefig(out_path)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    result = permutation_importance(classifier, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
    perm_sorted_idx = result.importances_mean.argsort()
    ax.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,
        labels=X_test.columns[perm_sorted_idx],
    )
    ax.set_title("Permutation Importances on features\n(test set)")
    ax.set_xlabel("Decrease in accuracy score")
    out_path = os.path.join(outdir, f"{output_name}_{run}_{classifier.__class__.__name__}_TEST.png")
    fig.savefig(out_path)



def save_statistical_report(accuracy, precision, recall, f1, output_name, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = folder + str(output_name) + ".txt"

    algs = ["lda_th", "lda", "rf", "xgboost", "lda_rf", "frbc_th", "frbc", "lda_frbc"]
    metrics = ["accuracy", "precision", "recall", "fscore"]

    f = open(path, "w")
    f.write("\t")
    f.write("\t")
    f.write("\t")
    for metric in metrics:
        f.write(str(metric))
        f.write("\t")
    f.write("\n")

    for i in range(len(algs)):
        f.write(str(algs[i]))
        f.write("\t")

        f.write("best")
        f.write("\t")
        f.write(str("{:.3f}".format(max(accuracy[i]))))
        f.write("\t")
        f.write(str("{:.3f}".format(max(precision[i]))))
        f.write("\t")
        f.write(str("{:.3f}".format(max(recall[i]))))
        f.write("\t")
        f.write(str("{:.3f}".format(max(f1[i]))))

        f.write("\n")
        f.write("\t")

        f.write("median")
        f.write("\t")
        f.write(str("{:.3f}".format(statistics.median_high(accuracy[i]))))
        f.write("\t")
        f.write(str("{:.3f}".format(statistics.median_high(precision[i]))))
        f.write("\t")
        f.write(str("{:.3f}".format(statistics.median_high(recall[i]))))
        f.write("\t")
        f.write(str("{:.3f}".format(statistics.median_high(f1[i]))))

        f.write("\n")
        f.write("\t")

        f.write("mean")
        f.write("\t")
        f.write(str("{:.3f}".format(statistics.mean(accuracy[i]))))
        f.write("\t")
        f.write(str("{:.3f}".format(statistics.mean(precision[i]))))
        f.write("\t")
        f.write(str("{:.3f}".format(statistics.mean(recall[i]))))
        f.write("\t")
        f.write(str("{:.3f}".format(statistics.mean(f1[i]))))

        f.write("\n")
        f.write("\t")

        f.write("stdev")
        f.write("\t")
        f.write(str("{:.3f}".format(statistics.pstdev(accuracy[i]))))
        f.write("\t")
        f.write(str("{:.3f}".format(statistics.pstdev(precision[i]))))
        f.write("\t")
        f.write(str("{:.3f}".format(statistics.pstdev(recall[i]))))
        f.write("\t")
        f.write(str("{:.3f}".format(statistics.pstdev(f1[i]))))

        f.write("\n")

        f.write("cm_best")
        f.write("\t")
        f.write(str(np.argmax(accuracy[i])))
        f.write("\t")
        f.write("cm_median")
        f.write("\t")
        f.write(str(np.argsort(accuracy[i])[len(accuracy[i]) // 2]))
        f.write("\n")
        f.write("\n")
    f.close()
