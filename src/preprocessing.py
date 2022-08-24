from typing import List
import pandas as pd
import numpy as np
from typing import List
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
import torch
import torchvision
import datetime
from PIL import Image

def _remove_targets(df:pd.DataFrame):
    _df = df.copy()
    _df["last.status"].replace({"discharged":0, "deceased":1}, inplace=True)
    targets = ["last.status", "is_icu", "was_ventilated", "length_of_stay"]

    df_targets = _df[targets]
    _df = _df.drop(targets + ["to_patient_id", "covid19_statuses",
     "visit_start_datetime", "invasive_vent_days"], axis=1)
    return _df, df_targets, targets

def process_features(df:pd.DataFrame, impute:bool=True, 
        remove_identifiers:bool=True, normalize_dates:bool=True,
        binary_mode:str="-1", numeric_mode:str="mean",
        remove_targets:bool=False) -> pd.DataFrame:
    """
    Main preprocessing routine for the dataset. 
    """
    _df = df.copy() # work on a copy

    #o, n, b = detect_types(_df)
    if remove_identifiers:
        _df.drop(["to_patient_id", "covid19_statuses"], axis=1, inplace=True)
    
    if normalize_dates: # dates are 
        _df["visit_start_datetime"] = pd.to_datetime(_df["visit_start_datetime"])-datetime.datetime(1901,1,1)
        _df["visit_start_datetime"] = _df["visit_start_datetime"].dt.days

    # the patients status is given as discharged/deceased
    _df["last.status"].replace({"discharged":0, "deceased":1}, inplace=True) 

    # fill NaNs that make sense
    na_zeros = ["invasive_vent_days", #days are NaN for patients not ventilated
                "kidney_replacement_therapy", "kidney_transplant", #either 1 or NaN in the whole dataset - treat as binary
                ]
    _df[na_zeros] = _df[na_zeros].fillna(0) 

    # manually estimated colums that can be One-Hot Encoded. 
    onehot = ["age.splits", "visit_concept_name", # those do not have missing data entries
        "gender_concept_name", "Urine.protein", "hf_ef_v", "smoking_status_v",
        "Other.anticoagulation.therapy"]

    enc = OneHotEncoder()
    enc.fit(_df[onehot])

    encoded = pd.DataFrame(enc.transform(_df[onehot]).toarray(),
            columns=enc.get_feature_names_out())

    _df = pd.concat([_df.drop(columns=onehot).reset_index(drop=True),
            encoded], axis=1)

    _df.replace({"No":0, "Yes":1,
                False:0, True:1,
                }, inplace=True)

    ### Imputation TODO: re-write routine for new imputation methods
    if impute:
        binary_columns, numeric_columns = infer_column_types(_df) # infer column types again

        _df = impute_numeric(_df, numeric_columns, mode=numeric_mode)
        impute_binary(_df, binary_columns, mode=binary_mode)
        
    if remove_targets:
        _df, df_targets, targets = _remove_targets(_df)
        return _df, df_targets, targets 
    else:
        return _df


def impute_binary(df:pd.DataFrame, binary_columns:List, mode:str="-1"):
    """
    Impute binary columns on a given DataFrame. Working on the DataFrame 
    itself.

    Parameters
    ----------
    df : Input DataFrame

    binary_columns : List columns to impute, naturally received from 
        infer_column_types

    mode : Mode used. Default "-1". More to be done TODO: add more
        - "-1" : default. Will fill NA's with -1 which does not interfere with
          data was available in the first place and gives the possibility to 
          refer back to missing Information.
    """

    if mode == "-1":
        df[binary_columns] = df[binary_columns].fillna(-1)
    elif mode == "0":
        df[binary_columns] = df[binary_columns].fillna(0)


def impute_numeric(df:pd.DataFrame, numeric_columns:List, mode:str="mean"):
    """
    Impute numeric columns on a given DataFrame. Working on the DataFrame 
    itself.

    Parameters
    ----------
    df : Input DataFrame

    binary_columns : List columns to impute, naturally received from 
        infer_column_types

    mode : Mode used. Default "mean". More to be done TODO: add more
        - "mean" : default. Will fill NA's with mean column value.
        - "median" : Fill NA's with median column value
        - "nn" : use k-NN
    """
    _df = df.copy()

    if mode == "mean":
        avg = df[numeric_columns].mean()
        for c in numeric_columns:
            _df[c].fillna(avg[c], inplace=True)
    elif mode == "median":
        avg = df[numeric_columns].mean()
        for c in numeric_columns:
            _df[c].fillna(avg[c], inplace=True)
    elif mode == "nn":
        _df = pd.DataFrame(KNNImputer().fit_transform(_df), columns=df.columns)
        
    elif mode == "zeros":
        _df[numeric_columns] = df[numeric_columns].fillna(0)
    return _df


def infer_column_types(df:pd.DataFrame) -> List:
    _df = pd.DataFrame(df.isna().sum())
    null_cols = _df[_df[0]>0].T.columns

    na_dict = {}
    for c in null_cols:
        na_dict[c] = dict(df[c].value_counts())

    binary_columns = []
    numeric_columns = []

    try:
        for k in na_dict.keys():
            # if we have 2 items in the values list we very most likely have a binary column, which we save
            if len(na_dict[k].values())==2: 
                binary_columns.append(k)
            elif len(na_dict[k].values())>2:
                numeric_columns.append(k)
        assert (set(na_dict.keys()) - set(binary_columns) - set(numeric_columns)) == set() # assert we did not miss any column

    except AssertionError as msg:
        print("Error infering column types", msg)

    return binary_columns, numeric_columns


def rename_cols(df:pd.DataFrame):
    """
    Unused for now. TODO: use or remove
    """
    rename_dict = {'to_patient_id':'patient_id', 'age.splits':'age',
       'gender_concept_name':'gender', 'visit_concept_name':'visit_type'}
    
    df.rename(columns=rename_dict, inplace=True)

def detect_types(df:pd.DataFrame) -> List:
    dtypes = dict(df.dtypes)

    objects = []
    numeric = []
    boolean = []

    for col, t in zip(dtypes.keys(), dtypes.values()):
        if t == "object" and "Bool" not in col:
            objects.append(col)
        elif t == "float64" or t == "int64":
            numeric.append(col)
        elif t == "bool" or "Bool" in col:
            boolean.append(col)

    return objects, numeric, boolean


def blend(origin, mask1=None, mask2=None):
    """
    Blend masks over image.
    """
    img = torchvision.transforms.functional.to_pil_image(origin + 0.5).convert("RGB")
    if mask1 is not None:
        mask1 =  torchvision.transforms.functional.to_pil_image(torch.cat([
            torch.zeros_like(origin),
            torch.stack([mask1.float()]),
            torch.zeros_like(origin)
        ]))
        img = Image.blend(img, mask1, 0.2)
        
    if mask2 is not None:
        mask2 =  torchvision.transforms.functional.to_pil_image(torch.cat([
            torch.stack([mask2.float()]),
            torch.zeros_like(origin),
            torch.zeros_like(origin)
        ]))
        img = Image.blend(img, mask2, 0.2)
    
    return img
