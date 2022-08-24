from ipaddress import collapse_addresses
import pandas as pd
import math
from typing import Dict, List
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier

def compute_bmi(weight, height) -> float:
    """
    Compute BMI as per CDC definition: BMI = weight (kg) / [height (m)]^2
    from:
    https://www.cdc.gov/healthyweight/assessing/bmi/adult_bmi/index.html

    Input provided as kg and cm
    """
    return weight/math.pow(height/100,2)

def create_patient(inputs:Dict, columns:List, cluster_means:pd.DataFrame=None):
    #print(inputs)
    #print(cluster)
    patient_df = pd.DataFrame(columns=columns, index=[0])
    #input_df = pd.DataFrame(inputs, index=[0])
    #print(input_df)
    for input in inputs.keys():
        if input in columns: # autofill
            patient_df[input] = inputs[input]
        elif input == "age": # escape age
            for col in patient_df.filter(regex="age").columns.tolist():
                if inputs["age"] in col:
                    patient_df[col] = 1
                else:
                    patient_df[col] = 0
        elif input == "gender": # escape gender
            for col in patient_df.filter(regex="gender_concept").columns.tolist():
                if str.upper(col).endswith(str.upper(inputs["gender"])):
                    if col.endswith("FEMALE") and inputs["gender"] == "Male":
                        patient_df[col] = 0
                    else:
                        patient_df[col] = 1
                else:
                    patient_df[col] = 0
        elif input == "BMI": # escape BMI
            patient_df["Body mass index (BMI) [Ratio]"] = round(inputs["BMI"], 2)
        elif input == "smoking_status":
            smoke_cols = patient_df.filter(regex="smoking_status").columns.tolist()
            for col in smoke_cols:
                if inputs["smoking_status"] in col:
                    patient_df[col] = 1
                elif inputs["smoking_status"] == "No Answer":
                    patient_df["smoking_status_v_nan"] = 1
                else:
                    patient_df[col] = 0
            patient_df[smoke_cols] = patient_df[smoke_cols].fillna(0) # INPLACE NEVER WORKS
        elif input == "heart_failure":
            if inputs["heart_failure"] == "CAD":
                inputs["heart_failure"] = "cad"
            hf_cols = patient_df.filter(regex="hf_ef|cad")
            #hf_cols = [s.lower() for s in hf_cols] # to lower
            for col in hf_cols:
                if inputs["heart_failure"] in col:
                    patient_df[col] = 1
                else:
                    patient_df[col] = 0
    # returns patient with parsed inputs and NaNs elsewhere
    return patient_df


def __create_patient(archetype:pd.Series, inputs:Dict) -> pd.Series:
    """
    Fill the archetype data with the user inputs.
    """

    _archetype = archetype.copy() # create a copy to work on

    # loop through dict key-values 
    for k,v in inputs.items():
        _current = _archetype.filter(regex=k) # select current keys values

        if k == "age":
            # age.splits_[18,59]
            for col in _current.index:
                #_lower = int(col[-6:-4])
                #_upper = int(col[-3:-1])
                if v == col[-7:]:
                    _archetype[col] = 1
                # if v in range(_lower, _upper):
                #     _archetype[col] = 1
                else:
                    _archetype[col] = 0

        elif k in ["gender", "smoking_status"]:
            for col in _current.index:
                if str.upper(col).endswith(str.upper(v)):
                    # if we have gender Male we also end with Male on Female, so we need to escape this
                    if col.endswith("FEMALE") and v == "Male":
                        _archetype[col] = 0
                    else:
                        _archetype[col] = 1
                else:
                    _archetype[col] = 0

        elif k == "BMI": # we hardcode here cuz lazy and straightforward
            _archetype["39156-5_Body mass index (BMI) [Ratio]"] = v
            if v > 30:
                _archetype["BMI.over30"] = 1
            # if one has over 35 bmi he will have 1 for both fields, so two ifs are fine here
            if v > 35:
                _archetype["BMI.over35"] = 1
    
    return _archetype