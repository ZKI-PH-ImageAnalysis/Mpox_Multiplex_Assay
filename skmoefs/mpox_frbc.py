from __future__ import print_function

import random
import os.path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from platypus.algorithms import *

from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from rcs import RCSInitializer, RCSVariator
from discretization.discretizer_base import fuzzyDiscretization
from toolbox import MPAES_RCS, load_dataset, normalize, is_object_present, store_object, load_object

warnings.filterwarnings('ignore')

def set_rng(seed):
    np.random.seed(seed)
    random.seed(seed)

datasets = ['df_seropos_high']
algs = ['moead'] 
#datasets = ['df_igg_all', 'df_seropos', 'df_seropos_high']
#algs = ['mpaes22', 'nsga3', 'moead', 'nsga2', 'gde3', 'ibea', 'spea2', 'epsmoea'] 

M = 200
Amin = 1
seed = 123
nEvals = 5000
capacity = 32
divisions = 8

variator = RCSVariator()
discretizer = fuzzyDiscretization(numSet=5)
initializer = RCSInitializer(discretizer=discretizer)

for dataset in datasets:
    set_rng(seed)
    X, y, attributes, inputs, outputs = load_dataset(dataset)
    X_n, y_n = normalize(X, y, attributes)
    X_train, X_test, y_train, y_test = train_test_split(X_n, y_n, test_size=0.3, random_state=seed)
    for alg in algs:
        mpaes_rcs_fdt = MPAES_RCS(M=M, Amin=Amin, capacity=capacity,
                                  divisions=divisions, variator=variator,
                                  initializer=initializer, moea_type=alg,
                                  objectives=['accuracy', 'trl'])
        mpaes_rcs_fdt.fit(X_train, y_train, max_evals=nEvals)
                
    classifier = mpaes_rcs_fdt.classifiers[0]
    scores = []
    
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print("FRBC train accuracy:", train_accuracy)
    print(classification_report(y_train, y_train_pred))
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("FRBC test accuracy:", test_accuracy)
    print(classification_report(y_test, y_test_pred))
    
    classifier.show_RB(inputs, outputs)
    
    mpaes_rcs_fdt.show_pareto()
    mpaes_rcs_fdt.show_pareto(X_test, y_test)
    mpaes_rcs_fdt.show_pareto_archives()
    mpaes_rcs_fdt.show_pareto_archives(X_test, y_test)
    #mpaes_rcs_fdt.show_model('median', inputs, outputs)
            
