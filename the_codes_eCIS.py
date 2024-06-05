import numpy as np
import matplotlib
import json
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from random import sample
from sklearn import svm, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,roc_curve, precision_recall_curve
import matplotlib.pyplot as plt 
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, precision_score
import pickle
from xgboost import XGBClassifier
def prepare_features(df, lst):
    """
    

    Parameters
    ----------
    df : dataframe of data
    lst : list of names of features

    Returns
    -------
    2D numpy array of features
    1D array of binary labels

    """
    return np.array(df[lst]), np.array(df["binary_label"])
def norm_features(features, lst):
    """
    

    Parameters
    ----------
    features : numpy array of features
    lst : list of names of features
    Returns
    -------
    norm_feat : min max normalised feature 2D numpy array

    """
    for i in lst:
        features[:,i] = features[:,i]/features[:,i].max(axis = 0)
    return features

def div_train_and_test(features, labels):
    """
    

    Parameters
    ----------
    features : 2D numpy array of ecis features
    labels : 1D numpy array of labels

    Returns
    -------
    x_train : 2D array of training features
    x_test : 2D array of testing features
    y_train : 1D array of training labels
    y_test : 1D array of testing labels
    

    """
    x_train, x_test, y_train, y_test = train_test_split(features, labels,stratify=labels)
    return x_train, x_test, y_train, y_test


def trainXGBoost(x_train, x_test, y_train, y_test, n_estimators = 30, max_depth = 7, min_child_weight = 20 ):
    """
    

    Parameters
    ----------
    x_train: the training set (features)
    y_train: the training set (labels)
    x_test: the test set (features)
    y_test: the test set (labels)
    n_estimators: the numbers of trees in the model (default 30)
    max_depth: the maximum possible number of levels in each tree (default 7)
    min_child_weight: the minimum possible number of samples in the leaf (default 20)
    
    Returns
    -------
    the trained classifier
    dataframe of accuracy, recall and precision for the individual training and testing 
    list of predictions of clf for the test set
    list of actual labels

    """
    clf = XGBClassifier(n_estimators = n_estimators , max_depth = max_depth, min_child_weight = min_child_weight)
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test) 
    d = pd.DataFrame(data=[accuracy_score(y_test, preds), recall_score(y_test, preds), precision_score(y_test, preds)], index=["accuracy", "recall", "precision"]) 
    return clf, d, list(preds), list(y_test)


def xgBoostClassifierKfoldWithFixedDp(x,y, fnames, dp = 8, sl = 10, nest = 10, folds = 5):
    """
    

    Parameters
    ----------
    x: features (np array)
    y: labels (np 1D array)
    fnames: features names
    nest: the numbers of trees in the model (default 30)
    dp: the maximum possible number of levels in each tree (default 7)
    sl: the minimum possible number of samples in the leaf (default 20)
    folds: number of folds for N-fold validation 
    Returns
    -------
    
    dictionary of accuracy, recall and precision for the each training and testing in cross validation step
    predicted_true_dict: dictionary of classifier results for each test sample

    """
    acc_rf = []
    recall_rf = []
    prec_rf = []
    number_of_k_fold = 1
    for i in range(number_of_k_fold):
     
       predicted_true_dict = {"predicted":[], "true":[], "proba": []}
       kfold = StratifiedKFold(folds, shuffle=True, random_state = None)
       for train, test in kfold.split(x,y):

           xd, yd = x[train], y[train]
           xt, yt = x[test], y[test]
          
           xg_params = trainXGBoost(xd, xt, yd, yt, n_estimators = nest,max_depth = dp, min_child_weight = sl )
           predicted_true_dict["predicted"].extend(xg_params[2])
           predicted_true_dict["true"].extend(xg_params[3])
           tmp = xg_params[0].predict_proba(xt)[:, 1]
           predicted_true_dict["proba"].extend(list(tmp))

       true_label = predicted_true_dict["true"]
       preds = predicted_true_dict["predicted"]
       acc_rf.append(accuracy_score(true_label, preds))
       recall_rf.append(recall_score(true_label, preds))
       prec_rf.append(precision_score(true_label, preds))

       return {"accuracy": acc_rf, "recall": recall_rf, "precision": prec_rf}, predicted_true_dict
def saveModel(model, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(model, f)

def openModel(file_name):
    with open(file_name, 'rb') as f:
        clf = pickle.load(f)
    return clf


#%%the script
input_features_file = "eCIS_features.csv"
features_df = pd.read_csv(input_features_file)
features_list = list(features_df.columns)
features_list.remove("gene_id")
features_list.remove("genome_id")
features_list.remove("label")
features_list.remove("protein_seq")
features_list.remove("binary_label")
x, y = prepare_features(features_df, features_list)
# xgboost chosen params analysis
features_list_disp = ['Length',
 'Signal Peptide Presence',
 'DUF4157 Adjacency',
 'DUF4157 Presence',
 'Cloud Domain Presence',
 'Distance from the Last Core Gene',
 'N-Term Hydrophobicity',
 'N-Term Molecular Weight',
 'N-Term Charge',
 'N-Term Disorder-promoting Index',
 'N-Term Aromaty']
metric_dict = xgBoostClassifierKfoldWithFixedDp(x,y,features_list_disp, sl = 4, nest = 40, folds = 5,dp = 7)
clf = XGBClassifier(n_estimators = 40 , max_depth = 7, min_child_weight = 4)
clf.fit(x, y)
model_variable = clf
file_name = "xg_boost_forest.pkl"
saveModel(model_variable, file_name)