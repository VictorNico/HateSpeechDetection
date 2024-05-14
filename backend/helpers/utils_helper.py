# importation de librairie

# module de manipulation des donnees
import pandas as pd
import matplotlib.pyplot as pl
import numpy as np

# module d'entrainement
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import precision_score,accuracy_score,f1_score, recall_score
#from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
import time
import os
import joblib

# module de pretraitement
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer
from itertools import combinations
from imblearn.over_sampling import SMOTE


def nominal_factor_encoding(data, variables_list=["Sexe"]):
    """Apply One Hot Encoding (OHE) on ordinal factor dimension
    Args:
      data: A dataframe containing the dimension to standardize
      variables_list: List of dimension on which apply the OHE

    Returns:
      The new dataframe with all dimension standardized.
    """
    dataframe = data.copy(deep=True)
    for col in variables_list:
        # print(f"{col} ---<>----- {dataframe[col].unique().tolist()}")
        dataframe[col] = dataframe[col].apply(lambda x: x +'__'+ col.replace(' ', '_'))
    (ohe,merge_ohe_col) = joblib.load("./models/ohe.sav")
    ohe_data = pd.DataFrame(ohe.transform(dataframe[variables_list]).toarray(), columns=merge_ohe_col) # make the one hot encoding and save the result inside a temp source
    dataframe = pd.concat([ohe_data, dataframe], axis=1) #  concat existing and news columns dimensions
    dataframe = dataframe.drop(variables_list, axis=1) # remove all nominal unencoded dimensions
    return (dataframe, ohe.categories_)

def numeric_uniform_standardization(data, variables_list):
    """Use max division standardize dimension
    Args:
      data: A dataframe containing the dimension to standardize
      variables_list: List of dimension on which apply the standardization

    Returns:
      The new dataframe with all dimension standardized.
    """

    dataframe = data.copy(deep=True)
    # 1) for each variable
    maxis = joblib.load('./models/uniformMax.sav')
    for var in variables_list:
        # get maximum value
        dataframe[var] =  dataframe[var].astype('float64')
        maxi = maxis[var]
        dataframe[var] = dataframe[var]/maxi
    return dataframe

def numeric_standardization_with_outliers(data, variables_list):
    """Use IQR to standardize dimension with extrem values
    Args:
      data: A dataframe containing the dimension to standardize
      variables_list: List of dimension on which apply the standardization aware outliers

    Returns:
      The new dataframe with all dimension standardized.
    """

    dataframe = data.copy(deep=True)
    # 1) for each variable
    maxis = joblib.load('./models/numerOutliersMax.sav')
    for var in variables_list:
        # a) compute Q1 and Q3
        dataframe[var] =  dataframe[var].astype('float64')
        (inf,sup) = maxis[var]
        for line in dataframe.index.values.tolist():
            # if less than inf
            if dataframe.loc[line, var] < inf:
                dataframe.loc[line, var] = inf/sup
            # else greater than sup
            elif dataframe.loc[line, var] > sup:
                dataframe.loc[line, var] = 1
            # else
            else:
                dataframe.loc[line, var] = dataframe.loc[line, var]/sup
    return dataframe

def preprocessing(dataframe):
    # calculons de nouvelle variables sur la base de l'existant
    # nous calculons la difference entre trim 2 et trim 1, trim 3 et trim 2, trim 3 et trim 1.
    dataframe["Trm_2-1"] = dataframe["Trm_2"] - dataframe["Trm_1"]
    dataframe["Trm_3-2"] = dataframe["Trm_3"] - dataframe["Trm_2"]
    dataframe["Trm_3-1"] = dataframe["Trm_3"] - dataframe["Trm_1"]

    # nous calculons aussi l'influence de l'age sur la moyenne d'un élève
    dataframe["ratio_moy_age"] = dataframe["Moy."] / dataframe["Age."]

    # chargement des configurations de colonnes
    cols = joblib.load("./models/cols.sav")
    # nous binarisons ici les variables catégorielles
    DATA_OHE, _ = nominal_factor_encoding(
        dataframe,
        cols['categorial_col']
        )

    # nous plongeons les donnees dans un interval 0,1
    DATA_OHE_LB_LBU_STDU = numeric_uniform_standardization(
        DATA_OHE,
        cols['numeric_uniform_colums']
        )
    # normalisation of numeric data with outliers to deeve it into interval 0,1
    DATA_OHE_LB_LBU_STDU_STDWO = numeric_standardization_with_outliers(
        DATA_OHE_LB_LBU_STDU,
        cols['numeric_with_outliers_columns']
        )

    return DATA_OHE_LB_LBU_STDU_STDWO

def predictor(dataframe):
#     print(dataframe)
    clf = joblib.load("./models/rfc_edutrack.sav")
    keys = joblib.load("./models/xtrain_cols.sav")
#     print(pd.DataFrame(data=clf.predict_proba(dataframe),columns=["0","1"]))
    y_pred = clf.predict(dataframe)
    vals = list(clf.feature_importances_)
    result_dict = dict(zip(keys, vals))
    data = dict(sorted(result_dict.items(), key=lambda x: abs(x[1]), reverse=False)[:])
    ret = {'prediction':[int(x) for x in list(y_pred)],'args':data, 'proba':pd.DataFrame(data=clf.predict_proba(dataframe),columns=["0","1"])}
#     print(ret)
    return ret