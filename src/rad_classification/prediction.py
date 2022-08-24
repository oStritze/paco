"""
This is basically the same as in src/ehd_classification/prediction.py, only without the imputation steps.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import auc, roc_curve, roc_auc_score, mean_absolute_error, balanced_accuracy_score, f1_score, accuracy_score, make_scorer, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as Pipeline_sampler
from imblearn.under_sampling import RandomUnderSampler

random_state = 42
rnd_state = random_state


def classify(X, y, k_fold=5,
            solver="ovo", rnd_state=42, class_weights=None,
            pca=False):

    clfs = [('rf', RandomForestClassifier(n_estimators=10, max_depth=5,
                random_state=rnd_state, class_weight=class_weights)),
            ('lreg', LogisticRegression(random_state=rnd_state, max_iter=250,
                class_weight=class_weights)),
            ('svm', SVC(probability=True, class_weight=class_weights, random_state=rnd_state)),
            ("xgboost", XGBClassifier(n_jobs=4, n_estimators=20, random_state=rnd_state)),
            ('MLP', MLPClassifier(solver="adam", learning_rate="adaptive",
                    random_state=rnd_state, max_iter=500, #class_weight=class_weights
                    )),
            ]
    param_grid = {}

    #fig, ax = plt.subplots(2, len(clfs), figsize=(20, 8))
    #fig.patch.set_facecolor('#ffffff')
    res = pd.DataFrame()

    scoring = { 'accuracy': make_scorer(accuracy_score),
                'balanced_accuracy': make_scorer(balanced_accuracy_score),
                #'f1_weighted': make_scorer(f1_score, average = 'weighted'),
                'roc_auc': make_scorer(roc_auc_score, multi_class=solver, needs_proba=True, average="weighted"),
                'recall': make_scorer(recall_score, average = 'micro'),
            }

    for i, c in enumerate(clfs):
        if not pca:
            pipe = Pipeline([('scaler', StandardScaler()),
                            #('pca', PCA(n_components=0.95)),
                            c])
        elif pca:
            pipe = Pipeline([('scaler', StandardScaler()),
                            ('pca', PCA(n_components=0.95)),
                            c])

        gscv = GridSearchCV(pipe, param_grid, scoring=scoring, n_jobs=4, cv=k_fold, refit="roc_auc")
        gscv.fit(X, y)

        _res = pd.DataFrame(gscv.cv_results_)
        #_res.filter(regex="mean_test")
        _res.index=[c[0]]
        res = pd.concat([res, _res])

        # for score_name, scorer in scoring.items():
        #     #print(score_name)
        #     res.at[c[0], f"train-{score_name}"] = scorer(gscv.best_estimator_, X, y)
        #     res.at[c[0], f"test-{score_name}"] = scorer(gscv.best_estimator_, X_val, y_val)


        #ConfusionMatrixDisplay.from_estimator(gscv.best_estimator_, X, y, ax=ax[0, i])
        #ax[0, i].set_title(f"{c[0]}")
        #ConfusionMatrixDisplay.from_estimator(gscv.best_estimator_, X_val, y_val, ax=ax[1, i])
    
    #fig.tight_layout()
    return res#, fig

def classify_imbalanced(X, y, k_fold=5,
            sampler=RandomUnderSampler(random_state=random_state),
            solver="ovo", rnd_state=42, class_weights=None,
            pca=False):

    clfs = [('rf', RandomForestClassifier(n_estimators=10, max_depth=5,
                random_state=rnd_state, class_weight=class_weights)),
            ('lreg', LogisticRegression(random_state=rnd_state, max_iter=250,
                class_weight=class_weights)),
            ('svm', SVC(probability=True, class_weight=class_weights,random_state=rnd_state)),
            ("xgboost", XGBClassifier(n_jobs=4, n_estimators=20, random_state=rnd_state)),
            ('MLP', MLPClassifier(solver="adam", learning_rate="adaptive",
                    random_state=rnd_state, max_iter=500, #class_weight=class_weights
                    )),
            ]
    param_grid = {}

    #fig, ax = plt.subplots(2, len(clfs), figsize=(20, 8))
    #fig.patch.set_facecolor('#ffffff')
    res = pd.DataFrame()

    scoring = { 'accuracy': make_scorer(accuracy_score),
                'balanced_accuracy': make_scorer(balanced_accuracy_score),
                #'f1_weighted': make_scorer(f1_score, average = 'weighted'),
                'roc_auc': make_scorer(roc_auc_score, multi_class=solver, needs_proba=True, average="weighted"),
                'recall': make_scorer(recall_score, average = 'micro'),
            }

    for i, c in enumerate(clfs):
        if not pca:
            pipe = Pipeline_sampler([('scaler', StandardScaler()),
                                ("sampler", sampler),
                                c])
        elif pca:
            pipe = Pipeline_sampler([('scaler', StandardScaler()),
                                ("sampler", sampler),
                                ('pca', PCA(n_components=0.95)),
                                c])

        gscv = GridSearchCV(pipe, param_grid, scoring=scoring, n_jobs=4, cv=k_fold, refit="roc_auc")
        gscv.fit(X, y)

        _res = pd.DataFrame(gscv.cv_results_)
        #_res.filter(regex="mean_test")
        _res.index=[c[0]]
        res = pd.concat([res, _res])

    #     for score_name, scorer in scoring.items():
    #         #print(score_name)
    #         res.at[c[0], f"train-{score_name}"] = scorer(gscv.best_estimator_, X, y)
    #         res.at[c[0], f"test-{score_name}"] = scorer(gscv.best_estimator_, X_val, y_val)


    #     ConfusionMatrixDisplay.from_estimator(gscv.best_estimator_, X, y, ax=ax[0, i])
    #     ax[0, i].set_title(f"{c[0]}")
    #     ConfusionMatrixDisplay.from_estimator(gscv.best_estimator_, X_val, y_val, ax=ax[1, i])
    
    # fig.tight_layout()
    return res#, fig