from __future__ import print_function

import os
import os.path
import warnings
from sklearn.inspection import permutation_importance


import numpy as np

from utils import *

from platypus.algorithms import *

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
    X_spox = spox_sets.iloc[:, 1:].copy()

    if norm == True:
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train[X_train.columns] = min_max_scaler.fit_transform(X_train)
        X_test[X_test.columns] = min_max_scaler.transform(X_test)
        X_spox[X_spox.columns] = min_max_scaler.transform(X_spox)

    lda = LinearDiscriminantAnalysis(n_components=n_att).fit(X_train, y_train)

    y_test_pred = lda.predict(X_test)
    y_train_pred = lda.predict(X_train)
    
    #export_feature_importance(X_train, y_train, X_test, y_test_pred, lda, feature_folder, output_name, run)

    save_lda_plot(X_train, y_train, X_test, y_test_pred, lda, LDA_folder, output_name, run)
    # save LDA dimension
        
    y_spox_prediction = lda.predict(X_spox)
    conf_degrees_spox = np.max(lda.predict_proba(X_spox), axis = 1)

    spox_out = X_spox.copy() # use copy so it does not change original df
    spox_out['panel_detail'] = y_spox_prediction
    spox_out['conf'] = conf_degrees_spox
    spox_out = spox_out[spox_out["conf"] > threshold_value]

    conf_degrees_test = np.max(lda.predict_proba(X_test), axis = 1)
    conf_degrees_train = np.max(lda.predict_proba(X_train), axis = 1)
    conf_y_real, conf_y_pred = threshold_usage(lda, X_test, y_test, y_test_pred, conf_degrees_test, threshold_value)
    
    if threshold_use == True and len(conf_y_real) > 0 and len(conf_y_pred) > 0:
        test = X_test
        test["conf_degree"] = conf_degrees_test
        new_test = test[test["conf_degree"] > threshold_value]
        
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
    
    else:   
        save_misclassified_data(
            X_test, y_test, y_test_pred, output_name, str(classifier), mis_folder, run, threshold_value, threshold_use
        )
        
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
        
        save_confusion_matrix(y_test, y_test_pred, output_name, str(classifier), cm_folder, run, classifier=lda)
            
    save_classified_general(
            X_train, X_test, y_train, y_train_pred, y_test, y_test_pred, output_name, str(classifier), class_folder, run, conf_degrees_train, conf_degrees_test, threshold_use
        )

    save_unknown_preds(spox_out, unknown_pred_folder, str(classifier), run, output_name)

    return accuracy, precision, recall, f1, y_test_pred, y_train_pred


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

    # export important features
    export_feature_importance(X_train, y_train, X_test, y_test_pred, rf, feature_folder, output_name, run)


    # prediction of unknown samples
    y_spox_prediction = rf.predict(X_spox)
    spox_out = X_spox.copy()
    spox_out['panel_detail'] = y_spox_prediction

    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average="macro")
    recall = recall_score(y_test, y_test_pred, average="macro")
    f1 = f1_score(y_test, y_test_pred, average="macro")

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

    save_confusion_matrix(y_test, y_test_pred, output_name, str(classifier), cm_folder, run, classifier=rf)

    save_misclassified_data(
        X_test, y_test, y_test_pred, output_name, str(classifier), mis_folder, run, threshold_value, threshold_use
    )
    
    save_classified_general(
            X_train, X_test, y_train, y_train_pred, y_test, y_test_pred, output_name, str(classifier), class_folder, run, None, None, False
        )
    
    save_unknown_preds(spox_out, unknown_pred_folder, str(classifier), run, output_name)


    return accuracy, precision, recall, f1, y_test_pred, y_train_pred


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
        
    # prediction of unknown samples
    y_spox_prediction = rf.predict(X_spox_lda)
    spox_out = X_spox.copy()
    spox_out['panel_detail'] = y_spox_prediction


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

    save_confusion_matrix(y_test, y_test_pred, output_name, str(classifier), cm_folder, run, classifier=rf)

    save_misclassified_data(
        X_test, y_test, y_test_pred, output_name, str(classifier), mis_folder, run, threshold_value, threshold_use
    )
    
    save_classified_general(
            X_train, X_test, y_train, y_train_pred, y_test, y_test_pred, output_name, str(classifier), class_folder, run, None, None, False
        )

    save_unknown_preds(spox_out, unknown_pred_folder, str(classifier), run, output_name)

    return accuracy, precision, recall, f1


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
    Spox, _, attributes_spox, inputs, outputs = load_dataset(dataset_spox, True) #unlabelled dataset is added

    X_train, y_train = normalize(X_train, y_train, attributes_train)
    X_test, y_test = normalize(X_test, y_test, attributes_test)
    Spox, _ = normalize(Spox, _, attributes_spox) #unlabelled dataset is added

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

    spox_pred = classifier.predict(Spox) #prediction for unlabelled data
    y_test_pred = classifier.predict(X_test)
    y_train_pred = classifier.predict(X_train)
    
    conf_degrees_spox = classifier.conf_degree(Spox)
    conf_degrees_test = classifier.conf_degree(X_test)
    conf_degrees_train = classifier.conf_degree(X_train)
    if threshold_use == True:
        conf_y_real, conf_y_pred = threshold_usage(classifier, X_test, y_test, y_test_pred, conf_degrees_test, threshold_value)
    
    #threshold implementation for spox data
    conf_degrees_spox = classifier.conf_degree(Spox)
    spox_out = spox_set.copy() # TODO Replace here
    spox_out['panel_detail'] = spox_pred
    if threshold_use == True:
        spox_out['conf'] = conf_degrees_spox
        spox_out = spox_out[spox_out["conf"] > threshold_value]    

    train, test = train_sets
    replace_panel(test)
    
    if threshold_use == True and len(conf_y_real) > 0 and len(conf_y_pred) > 0:
        test["conf_degree"] = conf_degrees_test
        
        new_test = test[test["conf_degree"] > threshold_value]
        
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
    
    else:    
        save_misclassified_data(
            test.iloc[:, :-1], test.iloc[:, -1], y_test_pred, output_name, str(classifier_name), mis_folder, run, threshold_value, threshold_use
        )
        
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
    
        save_confusion_matrix(y_test, y_test_pred, output_name, str(classifier_name), cm_folder, run, classifier=None)
        
    save_classified_general(
        train.iloc[:, :-1], test.iloc[:, :-1], train.iloc[:, -1], y_train_pred, test.iloc[:, -1], y_test_pred, output_name, str(classifier_name), class_folder, run, conf_degrees_train, conf_degrees_test, threshold_use
    )
        
    save_unknown_preds(spox_out, unknown_pred_folder, str(classifier_name), run, output_name) #save results for spox data

    directory = str(classifier_name)
    new_dir = os.path.join(rule_folder, directory)
    
    if os.path.exists(new_dir + "/") == False:
        os.mkdir(new_dir)
    path = new_dir + "/" + str(output_name) + "_" + str(run) + ".txt"

    file_ = open(path, "a")
    classifier.show_RB(inputs, outputs, f=file_)
    file_.close()

    return accuracy, precision, recall, f1


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
    Spox, _, attributes_spox, inputs, outputs = load_dataset(dataset_spox, True) #unlabelled dataset is added

    X_train, y_train = normalize(X_train, y_train, attributes_train)
    X_test, y_test = normalize(X_test, y_test, attributes_test)
    Spox, _ = normalize(Spox, _, attributes_spox) #unlabelled dataset is added

    lda = LinearDiscriminantAnalysis(n_components=n_att)

    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    Spox_lda = lda.transform(Spox)
    
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

    y_spox_pred = classifier.predict(Spox_lda)
    y_test_pred = classifier.predict(X_test_lda)
    y_train_pred = classifier.predict(X_train_lda)
    
    train, test = train_sets
    replace_panel(test)
    
    save_misclassified_data(
            test.iloc[:, :-1], test.iloc[:, -1], y_test_pred, output_name, str(classifier_name), mis_folder, run, None, False
        )
        
    save_classified_general(
            train.iloc[:, :-1], test.iloc[:, :-1], train.iloc[:, -1], y_train_pred, test.iloc[:, -1], y_test_pred, output_name, str(classifier_name), class_folder, run, None, None, False
        )
    
    #threshold implementation for spox data
    conf_degrees_spox = classifier.conf_degree(Spox_lda)
    spox_out = spox_set.copy()
    spox_out['panel_detail'] = y_spox_pred
    spox_out['conf'] = conf_degrees_spox  
    save_unknown_preds(spox_out, unknown_pred_folder, str(classifier_name), run, output_name) #save results for spox data

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

    save_confusion_matrix(y_test, y_test_pred, output_name, str(classifier_name), cm_folder, run, classifier=None)

    parent_dir = rule_folder
    directory = str(classifier_name)
    new_dir = os.path.join(parent_dir, directory)

    if os.path.exists(new_dir + "/") == False:
        os.mkdir(new_dir)
    path = new_dir + "/" + str(output_name) + "_" + str(run) + ".txt"

    file_ = open(path, "a")
    classifier.show_RB(inputs, outputs, f=file_)
    file_.close()

    return accuracy, precision, recall, f1

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

    if norm == True:
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train[X_train.columns] = min_max_scaler.fit_transform(X_train)
        X_test[X_test.columns] = min_max_scaler.transform(X_test)
        X_spox[X_spox.columns] = min_max_scaler.transform(X_spox)

    rf = GradientBoostingClassifier(
        n_estimators=n_est, max_depth=depth, random_state=seed
    ).fit(X_train, y_train)

    y_test_pred = rf.predict(X_test)
    y_train_pred = rf.predict(X_train)

    export_feature_importance(X_train, y_train, X_test, y_test_pred, rf, feature_folder, output_name, run)

    # prediction of unknown samples
    y_spox_prediction = rf.predict(X_spox)
    spox_out = X_spox.copy()
    spox_out['panel_detail'] = y_spox_prediction

    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average="macro")
    recall = recall_score(y_test, y_test_pred, average="macro")
    f1 = f1_score(y_test, y_test_pred, average="macro")

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

    save_confusion_matrix(y_test, y_test_pred, output_name, str(classifier), cm_folder, run, classifier=rf)

    save_misclassified_data(
        X_test, y_test, y_test_pred, output_name, str(classifier), mis_folder, run, threshold_value, threshold_use
    )
    
    save_classified_general(
            X_train, X_test, y_train, y_train_pred, y_test, y_test_pred, output_name, str(classifier), class_folder, run, None, None, False
        )
    
    save_unknown_preds(spox_out, unknown_pred_folder, str(classifier), run, output_name)

    return accuracy, precision, recall, f1, y_test_pred, y_train_pred