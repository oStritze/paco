import pandas as pd
import numpy as np
from typing import List
from sklearn import pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from src.ehd_classification.training import train, load
import utils
from src.preprocessing import process_features

from sklearn.pipeline import Pipeline

pretty_labels = utils.get_pretty_labels()


def _define_pipe(method:str="k-means", tsne:bool=True,
                scale:bool=True, random_state:int=42) -> Pipeline:
    """
    Helper function creating the pipeline
    """

    #pipe = Pipeline([]) #empty pipeline initialization does not work
    if scale:
        pipe=Pipeline([('scaler', StandardScaler())])
    if tsne: 
        pipe.steps.append(('tsne', TSNE(n_components=2, learning_rate="auto",
                            init="pca", random_state=random_state)))
    return pipe

def cluster(df, n_clusters:int=5, method:str="k-means", tsne:bool=True,
            scale:bool=True, random_state:int=42) -> List:
    """
    Compute classes using unsupervised Clustering. 
    
    Parameters
    ----------
    df : Input (preprocessed) DataFrame. Scaling can be done here in place.

    n_clusters : Define number of Clusters. Default 5 as per Notebook 
        exploration, yielded most promising results.

    method : Which method used. Default "k-means".  TODO: add more?
        - "k-means" : default. Use k-means.
        - "ward" : ward scaling.

    t-SNE : If t-SNE is used or not.
    
    scale : Use a Standardscaler or not prior to PCA/t-SNE.
    """
    classes = []
    pipe = _define_pipe(method, tsne, scale, random_state)
    model = None

    if method == "k-means":
        model = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
    elif method == "ward":
        model = AgglomerativeClustering(linkage="ward", n_clusters=n_clusters)
    else:
        raise NotImplementedError("No other methods implemented for now.")
    
    _X = pipe.fit_transform(df) # preproc as defined per params
    classes = model.fit_predict(_X) # predict clusters

    return classes, _X

def compute_cluster_metrics(df, clusters, targets):
    """
    Get DataFrame and clusters array with same length as DataFrame and compute
    metrics based on the targets defined. 
    TODO: mean is the cheapest option. Maybe add median or more sophisticated 
    (weighted stuff)
    """

    try:
        assert(df.shape[0] == len(clusters))
    except:
        print("Dimension missmatch!")
    _df = df.copy()
    _df["cluster"] = clusters
    
    cluster_means = _df.groupby("cluster").mean().round(4)
    return cluster_means


def plot_cluster_barplot(targets:List, df:pd.DataFrame, df_targets:pd.DataFrame,
                        classes:List, n_clusters:int=5):
    
    _df = df.copy()
    # prepare result dataframe with multi-index
    tuples = []
    for t in targets:
        for i in np.unique(classes):
            tuples.append((t,i))

    index = pd.MultiIndex.from_tuples(tuples, names=["target", "cluster"])
    res = pd.DataFrame(index=index)

    # add classes
    _df["cluster"] = classes
    # add targets to the df since they were seperated
    _df = pd.concat([_df, df_targets], axis=1) 
    for target in targets:
        cnts = _df[["cluster", target]].value_counts().reset_index()
        if target != "length_of_stay":
            for i in np.unique(classes):
                res.loc[(target,i), "neg"] = cnts[(cnts["cluster"] == i) & (cnts[target] == 0)][0].iloc[0]
                res.loc[(target,i), "pos"] = cnts[(cnts["cluster"] == i) & (cnts[target] == 1)][0].iloc[0]
                res.loc[(target,i), "n"] = cnts[cnts["cluster"] == i][0].sum()
                #print(f"cluster {i}:", round(_pos / _sum, 3))
        else:
            for i in np.unique(classes):
                res.loc[(target,i), "n"] = cnts[cnts["cluster"] == i][0].sum()
                res.loc[(target,i), "avg_length_of_stay"] = round(
                        _df[_df["cluster"]==i][["length_of_stay"]].mean().iloc[0],
                    3)

    res["posrate"] = round(res["pos"] / res["n"], 3)

    plot_res = res.reset_index()

    #fig, ax = plt.subplots(2, 1, figsize=(8,10))
    fig = plt.figure(constrained_layout=True, figsize=(16,8))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 4, figure=fig)
    ax1 = fig.add_subplot(gs[0, :3])
    ax2 = fig.add_subplot(gs[0, 3])

    sns.set_theme(style="whitegrid")
    sns.barplot(data=plot_res[plot_res["target"] != "length_of_stay"],
                y="posrate", x="target",
                hue="cluster", #orient="v",
                palette=sns.color_palette()[:n_clusters],
                ax=ax1
                )
    ax1.set_ylim(0,1)
    ax1.legend(loc="upper left").set_title("Cluster")
    sns.barplot(data=plot_res[plot_res["target"] == "length_of_stay"],
                y="avg_length_of_stay", x="target",
                hue="cluster", #orient="h",
                palette=sns.color_palette()[:n_clusters],
                ax=ax2
                )
    ax2.legend([],[],frameon=False)

    return fig

def compute_target_statistics_per_cluster(targets:List, df:pd.DataFrame,
                    df_targets:pd.DataFrame, classes:List, length_scale:int=30):
    
    _df = df.copy()
    targets = targets + ["length_of_stay"]
    # prepare result dataframe with multi-index
    tuples = []
    for t in targets:
        for i in np.unique(classes):
            tuples.append((t,i))

    index = pd.MultiIndex.from_tuples(tuples, names=["target", "cluster"])
    res = pd.DataFrame(index=index)

    # add classes
    print(_df.shape, classes.shape, df_targets.shape)
    print(classes)
    _df["cluster"] = classes
    #print(_df.shape, classes.shape, df_targets.shape)

    # add targets to the df since they were seperated
    #_df = pd.concat([_df, df_targets], axis=1) 
    for target in targets:
        #print(_df)
        cnts = _df[["cluster", target]].value_counts().reset_index()
        if target != "length_of_stay":
            for i in np.unique(classes):
                res.loc[(target,i), "neg"] = cnts[(cnts["cluster"] == i) & (cnts[target] == 0)][0].iloc[0]
                res.loc[(target,i), "pos"] = cnts[(cnts["cluster"] == i) & (cnts[target] == 1)][0].iloc[0]
                res.loc[(target,i), "n"] = cnts[cnts["cluster"] == i][0].sum()
                #print(f"cluster {i}:", round(_pos / _sum, 3))
        else:
            for i in np.unique(classes):
                res.loc[(target,i), "n"] = cnts[cnts["cluster"] == i][0].sum()
                res.loc[(target,i), "avg_length_of_stay"] = round(
                        _df[_df["cluster"]==i][["length_of_stay"]].mean().iloc[0],
                    3)

    res["posrate"] = round(res["pos"] / res["n"], 3)

    plot_res = res.reset_index()

    # put all results in a column to make it easier to vis all in one place
    def _f(row):
        if row.target != "length_of_stay":
            val = row.posrate
        else:
            # because of the different value range of the stay length, we need 
            # scale that by length_scale, by default 30 so "months"
            val = row.avg_length_of_stay/length_scale
        return val
    plot_res["value"] = plot_res.apply(_f, axis=1)

    # add pretty labels TODO: make this optional or work on it in all cases
    pretty_labels["length_of_stay"] = f"Length of stay in days/{length_scale}"

    plot_res["pretty_target"] = plot_res.apply(lambda row: pretty_labels[row.target], axis=1)
    print(plot_res)
    return plot_res

def compute_clustering_metrics(X, clusters, settings):
    sil = metrics.silhouette_score(X, clusters)
    ch = metrics.calinski_harabasz_score(X, clusters)
    settings_str = f"Imputation: {settings['imputation']} - {settings['method']}, k={settings['n_clusters']}, seed={settings['seed']}"
    #print(type(settings), settings, settings_str)

    results = pd.DataFrame.from_dict({"Settings": [settings_str], "Silhouette": [sil], "Calinski-Harabasz": [ch]})
    # results_df = results_df.append({"Settings": settings_str, #TODO: prettify
    #     "Silhouette": sil, "Calinski-Harabasz": ch},
    #     ignore_index=True)
    return results
