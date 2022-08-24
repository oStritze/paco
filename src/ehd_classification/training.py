import typing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import auc, roc_curve, roc_auc_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import LeaveOneOut, GridSearchCV
import joblib
import matplotlib.pyplot as plt
import os

def train(X:pd.DataFrame, targets:pd.DataFrame, cv_k:int=10, n_jobs=8,
            path_prefix:str="", use_loo:bool=False, plot_cm:bool=False):
    pipe = Pipeline(
        [
            # the reduce_dim stage is populated by the param_grid
            ('scaler', MinMaxScaler()),
            ('rf', RandomForestClassifier(random_state=42))
        ]
    )

    param_grid = {
        'rf__n_estimators': [50, 100, 500],
        'rf__criterion': ['gini', 'entropy'],
        'rf__max_depth': [None, 25, 50],
        'rf__min_samples_leaf': [2, 4],
        #'max_features': ['sqrt', 'log2'],
        #'bootstrap': [True, False],
        #'random_state': [42]
    }

    loo = LeaveOneOut()
    
    print(f"saving models under: models/ehd/{path_prefix}")

    for target in targets.columns:
        y = targets[target]

        if target != "length_of_stay":
            print(f"fitting {target}... ")
            if use_loo:
                cv = GridSearchCV(pipe, param_grid=param_grid, cv=loo,
                    n_jobs=8, verbose=1,
                    )
                cv.fit(X, y)
                best_pipe = cv.best_estimator_
                print("ROC-AUC: ", roc_auc_score(best_pipe.predict(X), y)) 
                if plot_cm:
                    ConfusionMatrixDisplay.from_estimator(best_pipe, X, y) 

            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                cv = GridSearchCV(pipe, param_grid=param_grid, cv=cv_k, #cv=loo,
                                n_jobs=n_jobs, verbose=1,
                                )

                cv.fit(X_train, y_train)
                best_pipe = cv.best_estimator_

                print("Train ROC-AUC: ", roc_auc_score(best_pipe.predict(X_train), y_train))
                print("Test ROC-AUC: ", round(roc_auc_score(best_pipe.predict(X_test), y_test), 3))
                if plot_cm:
                    fig, ax = plt.subplots(1,2)
                    ConfusionMatrixDisplay.from_estimator(best_pipe, X_train, y_train, ax=ax[0])
                    ConfusionMatrixDisplay.from_estimator(best_pipe, X_test, y_test, ax=ax[1])

            joblib.dump(best_pipe, f"models/ehd/{path_prefix}{target}.pkl")


def load(file_prefix:str) -> list:
    models = []
    fnames = []

    print("trying to load models under: models/ehd/{file_prefix}")
    files = os.listdir("models/ehd")
    for f in files:
        if f.startswith(f"{file_prefix}"):
            models.append(joblib.load(f"models/ehd/{f}"))
            fnames.append(f)
    
    return models, fnames
    