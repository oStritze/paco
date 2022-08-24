import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import auc, roc_curve, roc_auc_score, mean_absolute_error, balanced_accuracy_score, f1_score, accuracy_score, make_scorer, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.impute import KNNImputer
from imblearn.pipeline import Pipeline as Pipeline_sampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

random_state = 42
rnd_state = random_state


def classify(X, y, imputer=KNNImputer(), k_fold=5,
            solver="ovo", rnd_state=42, class_weights=None,
            pca=False, groups=None):

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
                            ('imp', imputer),
                            ('scaler2', StandardScaler()),
                            #('pca', PCA(n_components=0.95)),
                            c])
        elif pca:
            pipe = Pipeline([('scaler', StandardScaler()),
                            ('imp', imputer),
                            ('scaler2', StandardScaler()),
                            ('pca', PCA(n_components=0.95)),
                            c])

        if groups is not None:
            splits = k_fold.split(X, y, groups) # using GroupKFold for merged data
            gscv = GridSearchCV(pipe, param_grid, scoring=scoring, n_jobs=4, cv=splits, refit="roc_auc")
        else:
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

def classify_imbalanced(X, y, imputer=KNNImputer(), k_fold=5,
            sampler=RandomUnderSampler(random_state=random_state),
            solver="ovo", rnd_state=42, class_weights=None,
            pca=False, groups=None):

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
                                ('imp', imputer),
                                ('scaler2', StandardScaler()),
                                ("sampler", sampler),
                                c])
        elif pca:
            pipe = Pipeline_sampler([('scaler', StandardScaler()),
                                ('imp', imputer),
                                ('scaler2', StandardScaler()),
                                ("sampler", sampler),
                                ('pca', PCA(n_components=0.95)),
                                c])

        if groups is not None:
            splits = k_fold.split(X, y, groups) # using GroupKFold for merged data
            gscv = GridSearchCV(pipe, param_grid, scoring=scoring, n_jobs=4, cv=splits, refit="roc_auc")
        else:
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

def compute_metrics(model, X_train, y_train, X_val, y_val, index=0):
    _targets = [0,1,2,3,4,6,7]
    
    model = model.fit(X_train, y_train)
    y_hat = model.predict(X_val)
    cm = confusion_matrix(y_val, model.predict(X_val))
    cm_pretty = pd.DataFrame(cm, index=_targets, columns=_targets)

    results = pd.DataFrame()
    results.at[index, "F1"] = f1_score(y_val, y_hat, average="weighted")
    results.at[index, "Accuracy"] = accuracy_score(y_val, y_hat)
    results.at[index, "Balanced Accuracy"] = balanced_accuracy_score(y_val, y_hat)
    results.at[index, "Recall"] = recall_score(y_val, y_hat, average="weighted")
    results.at[index, "ROC-AUC"] = roc_auc_score(y_val, model.predict_proba(X_val), multi_class="ovo", average="weighted")

    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    ACC = (TP+TN)/(TP+FP+FN+TN)

    results_per_outcome = pd.DataFrame(index=_targets)
    results_per_outcome["TPR-Sensitivity (recall)"] = TPR
    results_per_outcome["TNR-Specificity"] = TNR
    #print(f"TPR - Sensitivity: {TPR}")
    #print(f"TNR - Specificity: {TNR}")

    return results.round(3), results_per_outcome.round(3).T, cm_pretty