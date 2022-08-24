from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import streamlit as st
import os, sys, logging, time, math
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score, make_scorer, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import process_features
import utils
from clustering import compute_cluster_metrics, cluster, compute_target_statistics_per_cluster, compute_clustering_metrics
from preprocessing import process_features
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from src.data.lungdataset import LungData
import numpy as np
from streamlit_tabs.medical import upload, plot_preds_bar_echart, set_constants
from streamlit_tabs.general import plot_preds_echart, predictions, _form_patient_inputs
from ehd_classification.utils import compute_bmi, create_patient
from streamlit_echarts import st_echarts, st_pyecharts
from pyecharts import options as opts
from pyecharts.charts import Bar, Pie
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression as LR

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

pretty_labels = utils.get_pretty_labels()
sns.set_style("whitegrid")


def prepare_constants(ehd, clusters):
    _st = time.time()
    model = joblib.load("models/ehd/svm_roc_auc_trained_on_all.pkl")
    st.session_state["model"] = model

    st.session_state["kn_class"] = KNeighborsClassifier(n_neighbors=5)
    st.session_state["scaler"] = StandardScaler()
    st.session_state["lr"] = LR(class_weight="balanced", random_state=42)

    st.session_state["input-columns"] = ['kidney_replacement_therapy', 'kidney_transplant', 'htn_v', 'dm_v', 'cad_v',
                'malignancies_v', 'copd_v', 'Body mass index (BMI) [Ratio]',
                'age.splits_[18,59]', 'age.splits_[59,74]', 'age.splits_[74,90]',
                'gender_concept_name_FEMALE', 'gender_concept_name_MALE',
                'gender_concept_name_nan', 'hf_ef_v_HFpEF', 'hf_ef_v_HFrEF', 'hf_ef_v_No',
                'hf_ef_v_nan', 'smoking_status_v_Current', 'smoking_status_v_Former',
                'smoking_status_v_Never', 'smoking_status_v_nan']
    st.session_state["pretty-input-columns"] = ["Kidney Repl. Therapy", "Kidney Transplant","High Blood Pressure",
                "Diabetes","Coronary Artery Disease","Malignancies","COPD","BMI","Age 18-59","Age 59-74","Age 74-90",
                "Gender Female","Gender Male","Gender not specified/non binary","HFpEF","HFrEF","No Heart Failures","Heart Failures not specified",
                "Smoking - Current","Smoking - Former","Smoking - Never","Smoking - Not specified"]

    #print("PREPARE CONSTANTS")
    _ehd = ehd.copy()
    _ehd["cluster"] = clusters

    if "scaler" in st.session_state:
        sclr = st.session_state["scaler"]
    else:
        st.session_state["scaler"] = StandardScaler()
        sclr = st.session_state["scaler"]

    if "kn_class" in st.session_state:
        knc = st.session_state["kn_class"]
    else:
        st.session_state["kn_class"] = KNeighborsClassifier(n_neighbors=5)
        knc = st.session_state["kn_class"]

    # fit scaler
    if "input-columns" not in st.session_state:
        st.session_state["input-columns"] = ['kidney_replacement_therapy', 'kidney_transplant', 'htn_v', 'dm_v', 'cad_v',
                'malignancies_v', 'copd_v', 'Body mass index (BMI) [Ratio]',
                'age.splits_[18,59]', 'age.splits_[59,74]', 'age.splits_[74,90]',
                'gender_concept_name_FEMALE', 'gender_concept_name_MALE',
                'gender_concept_name_nan', 'hf_ef_v_HFpEF', 'hf_ef_v_HFrEF', 'hf_ef_v_No',
                'hf_ef_v_nan', 'smoking_status_v_Current', 'smoking_status_v_Former',
                'smoking_status_v_Never', 'smoking_status_v_nan']
    elif "input-columns" in st.session_state:
        sclr.fit(_ehd[st.session_state["input-columns"]])
        st.session_state["scaler"] = sclr # save back

        knc.fit(sclr.transform(_ehd[st.session_state["input-columns"]]), clusters)
        st.session_state["kn_class"] = knc

    st.session_state["manual-prediction"] = pd.DataFrame()

    logging.info("Running {} took: {}".format("prepare_constants", round(time.time()-_st, 3)))




@st.experimental_memo
def initialize_cluster_results():
    cluster_results_df = pd.DataFrame(columns=["Settings", "Silhouette", "Calinski-Harabasz"])
    return cluster_results_df

@st.cache
def _define_colormaps(colormap=px.colors.qualitative.Set1, n_clusters:int=5):
    cbar = colormap
    cmap_match = {} # matching colors to cluster names
    for i, c in zip(range(n_clusters),cbar):
        cmap_match[str(i)] = c
    return cbar, cmap_match

@st.cache
def _form_inputs(imputation, method, n_clusters, seed):
    """
    Takes raw form inputs and returns a dict that will be easily iterable later
    """
    inputs = {
        "imputation": imputation, "method": method,
        "n_clusters": n_clusters, "seed": seed
    }
    return inputs


@st.experimental_memo
def dummies(_prettycols):
    _dummies = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    preds = pd.DataFrame(_dummies, columns=_prettycols)
    return preds


@st.experimental_memo
def convert_to_tsne(df):
    return TSNE(n_components=2, perplexity=10, learning_rate=500, init="pca", random_state=42).fit_transform(df)

@st.cache
def _cluster(ld:LungData, df_pp, inputs):
    # df_pp = ld.process_features(numeric_mode=inputs["imputation"], 
    #         binary_mode=inputs["imputation"], 
    #         return_df = True)
    logging.info("Clustering with settings: {}. \nDF shape:{}".format(inputs, df_pp.shape))
    _st = time.time()

    classes, clustered_df = ld.cluster(df_pp, inputs["method"], n_clusters=inputs["n_clusters"])
    logging.info("Clustering took {}s".format(round(time.time()-_st, 3)))

    return classes, clustered_df


## TRY SCATTER AND COLOR PER LABELS
def echart_scatterplot(df, classes, n_clusters):
    data = [[round(x,3), round(y,3)] for x, y in zip(df[0], df[1])]
    pieces = []
    colors = _define_colormaps(n_clusters=n_clusters)[0]
    #print(data)
    #print(colors)
    for c in range(0, n_clusters):
        pieces.append({
            "value": c,
            "label": f'cluster {c}',
            "color": colors[c]
        })

    options = {
        "dataset": [
            {
                "source": data
            },
            {
            "transform": {
                "type": 'ecStat:clustering',
                #// print: true,
                "config": {
                "clusterCount": n_clusters,
                "outputType": 'single',
                "outputClusterIndexDimension": 2#DIENSIION_CLUSTER_INDEX
                }
            }
            }
        ],
        "tooltip": {
            "position": 'top'
        },
        "visualMap": {
            "type": 'piecewise',
            "top": 'middle',
            "min": 0,
            "max": n_clusters,
            "left": 10,
            "splitNumber": n_clusters,
            "dimension": 2,#DIENSIION_CLUSTER_INDEX,
            "pieces": pieces
        },
        "grid": {
            "left": 120
        },
        "xAxis": {},
        "yAxis": {},
        "series": {
            "type": 'scatter',
            "encode": { "tooltip": [0, 1] },
            "symbolSize": 15,
            "itemStyle": {
                "borderColor": '#555'
            },
            "datasetIndex": 1
        }
    }
    return options

#@st.experimental_memo
def _scatterplot_plotly(df_pp, classes, n_clusters):
    cbar, cmap_match = _define_colormaps(n_clusters=n_clusters)
    df = pd.DataFrame(df_pp)
    df["Cluster"] = classes.astype(str)
    df = df.sort_values(by="Cluster")
    fig = px.scatter(df,
                x=0, y=1, 
                color="Cluster", 
                #color_discrete_map=cmap_match,
                color_discrete_sequence=cbar[:n_clusters],
                #symbol=df_targets[target],
                #symbol_sequence=[0,4],
                labels={"color":"Cluster", "symbol":"Deceased"}
                #marker_symbol=[0,1]
                #size='petal_length',
                #hover_data=['petal_width'] 
                )
    # fig.update_traces(marker_size=10, opacity=0.75, 
    #             line=dict(
    #                 color='MediumPurple',
    #                 width=20,
    #             ),
    #     )
    fig.for_each_trace(lambda t: t.update({"marker":{"size":10, "opacity":0.7, #"symbol":"hexagon",
                                        "line":{"width":0.2, "color":"DarkSlateGrey"}
                                        }}
                                    ))
    fig.update_layout(
        plot_bgcolor="#f8f8f8",
        margin=dict(
            l=0,
            r=0,
            b=30,
            t=0,
            pad=4
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-.09,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.7)",
        ),
        xaxis= {
            #'range': [0.2, 1],
            'showgrid': True, # thin lines in the background
            'zeroline': False, # thick line at x=0
            'visible': False,  # numbers below
        }, 
        yaxis= {
            #'range': [0.2, 1],
            'showgrid': True, # thin lines in the background
            'zeroline': False, # thick line at x=0
            'visible': False,  # numbers below
        },
    )
    return fig

def clustering_get_top_feats(data, classes, n_feats=15):
    #lr = LR(class_weight="balanced", random_state=42)
    st.session_state["lr"] = LR(class_weight="balanced", random_state=42)
    if "lr" in st.session_state:
        _lr = st.session_state["lr"]
    else:
        st.session_state["lr"] = LR(class_weight="balanced", random_state=42)
    _lr.fit(data, classes)
    coef = pd.DataFrame(_lr.coef_, columns=data.columns).T
    #topfeats = coef.T.abs().max().sort_values(ascending=False)[:n_feats].index
    topfeats = coef.T.mean().sort_values(ascending=False)[:n_feats].index.tolist()
    return topfeats[::-1] # flip array for nicer presentation

def prepare_echart_radar_data(data):
    legend = [f"Cluster {i}" for i in data.index.values]
    indicator = []
    for col in data.columns:
        indicator.append({"name": col, "max":1})
    _data = []
    for i in range(1, len(data.index)+1):
        _data.append(
            {"value": data.iloc[i-1].round(3).values.tolist(), "name": f"Cluster {i}"}
        )
    
    return legend, indicator, _data

def echart_radar(data, legend, indicator):
    options = {
        "title": {
            "text": ''
        },
        "legend": {
            "bottom": "75%",
            "left": "0%",
            "orient": "vertical" ,
            "data": legend
        },
        "radar": {
            # "shape": 'circle',
            "radius": "80%",
            "indicator": indicator
        },
        #"radiusAxis" : {"max": 0.1},
        "tooltip": {"trigger": "item",
                 "axisPointer": { "type": 'cross', "crossStyle": { "type": 'solid' } }
        },
        "series": [
            {
            "type": 'radar',
            "data": data,
            "areaStyle": {"opacity":0.1},
            }
        ]
    }
    return options


def _barplot_plotly(X, target):
    #cbar, cmap_match = _define_colormaps(n_clusters=1)
    fig = px.bar(X.round(4), x="Settings", y=target,
            #color=target
            #color_discrete_map=cbar,
            )
    fig.update_layout(margin=dict(
            l=0,
            r=0,
            b=30,
            t=0,
            pad=4
        ),
        #autosize=False,
        height=200,
        xaxis= {
            #'visible': False
            'ticks' : "inside",
            'showticklabels': False,
            }
    )
    fig.update_traces(marker_color='#ff8c85')
    return fig


def _barplot_twoy_plotly(X):
    fig = go.Figure(
        data=[
            go.Bar(name='Silhouette', x=X["Settings"], y=X["Silhouette"], yaxis='y', offsetgroup=1),
            go.Bar(name='Calinski-Harabasz', x=X["Settings"], y=X["Calinski-Harabasz"], yaxis='y2', offsetgroup=2)
        ],
        layout={
            'yaxis': {'title': 'Silhouette'},
            'yaxis2': {'title': 'Calinski-Harabasz', 'overlaying': 'y', 'side': 'right'}
        }
    )
    # Change the bar mode and legend layout
    fig.update_layout(barmode='group',
                    legend=dict(yanchor="bottom",
                            y=0.01, 
                            xanchor="left", 
                            x=0.01, #orientation="h"
                            bgcolor="rgba(255,255,255,0.7)"),
                    margin=dict(
                        l=0,
                        r=0,
                        b=30,
                        t=0,
                        pad=4
                    ),
                    #autosize=False,
                    height=430,
                    xaxis= {
                        #'visible': False
                        'ticks' : "inside",
                        'showticklabels': True,
                        }
    )
    fig.update_layout(
        modebar_remove=['zoom', 'pan', 'select', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale', 'displaylogo']
    )
    return fig

@st.experimental_memo
def _compute_cluster_target_stats(targets, df, df_targets, classes, length_scale=30):
    return compute_target_statistics_per_cluster(targets, df, df_targets, classes, length_scale)


def build_imputation():
    return 0

def plot_imputation_examples(_ehd, col1, col2, col3, _height=2.5):
    feats = ["Lymphocytes [#/volume] in Blood by Automated count",
            "Creatine kinase [Enzymatic activity/volume] in Serum or Plasma",
            "vomiting_v"]

    with col1:
        f1 = st.selectbox("Check for Imputation Influences in...", options=_ehd.columns, key="f1",
                        index = int(np.where(_ehd.columns == feats[0])[0][0])
                        )
        fig1, ax = plt.subplots(figsize=(6, _height))
        sns.histplot(data=_ehd, x=f1,
                kde=True, bins=25,
                ax=ax
                )
        st.pyplot(fig1)
    with col2:
        f2 = st.selectbox("", options=_ehd.columns, key="f2",
                        index = int(np.where(_ehd.columns == feats[1])[0][0])
                        )
        fig2, ax = plt.subplots(figsize=(6, _height))
        sns.histplot(data=_ehd, x=f2, 
                kde=True, bins=25,
                ax=ax
                )
        st.pyplot(fig2)
    with col3:
        f3 = st.selectbox("", options=_ehd.columns, key="f3",
                        index = int(np.where(_ehd.columns == feats[2])[0][0])
                        )
        fig3, ax = plt.subplots(figsize=(6, _height))
        sns.histplot(data=_ehd, x=f3, 
                kde=True, bins=25,
                ax=ax
                )
        st.pyplot(fig3)

def create_cluster_means(ehd, classes):
    _ehd = ehd.copy()
    _ehd["Cluster"] = classes
    cluster_means = _ehd.groupby("Cluster").mean()
    return cluster_means



#@st.experimental_memo
def _compute_clustering_metrics(clustering_results, clustered_df, classes, inputs):
    results = compute_clustering_metrics(clustered_df, classes, inputs)
    return clustering_results.append(results).reset_index(drop=True)


@st.experimental_memo
def update_clustering_results(df):
    st.session_state["clustering_results"] = df



@st.experimental_memo
def _preproc_ehd(_ld, imputation, nn, mi_val):
    """
    Just a preproc wrapper function to use the streamlit memorize ability.
    """
    return _ld.process_features(binary_mode=imputation, numeric_mode=imputation,
                            k_nearest=nn, mi_val=mi_val, return_df=True, normalize_dates=False)


@st.experimental_memo
def create_patient_record(inputs:dict, archetype:pd.DataFrame()):
    """
    Create patient record from archetype and input.
    """
    return create_patient(archetype, inputs)

    
def _plot_scatter_overview(X, x, y, outcome):
    cmap = {"Hospitalized (only)": "#edf8e9", "ICU": "#fc8d62", "Ventilated": "#8da0cb", "Deceased":"#e78ac3",
            "ICU + Ventilated": "#c4644f", "Deceased + Ventilated": "#b98ac1", "Deceased + ICU":"#c35065",
            "Deceased + ICU + Ventilated":"#941e67"}
    fig = px.scatter(X,
                x=x, y=y, color=outcome,
                color_discrete_map=cmap,
    )
    fig.for_each_trace(lambda t: t.update({"marker":{"size":9, "opacity":0.7, #"symbol":"hexagon",
                                    "line":{"width":1, "color":"DarkSlateGrey"}
                                    }}
                                ))
    fig.update_layout(
        legend=dict(
                orientation="h",
                #yanchor="top",
                #y=-.17,
                y=1.2,
        #        xanchor="right",
        #        x=0.9,#1.0,
            ),
            legend_title_text="",
            margin=dict(
                l=0,
                r=0,
                b=30,
                t=0,
                pad=4
            ),
            height=380,
            xaxis= {
                #'visible': False
                'ticks' : "inside",
                'showticklabels': True,
            }
    )
    #fig.update_layout(
    #    modebar_remove=['zoom', 'pan', 'select', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale', 'displaylogo']
    #)
    return fig 

def _plot_heatmap_targets(X):
    trace = go.Heatmap(z=X, x=X.columns.values.tolist(), 
                y=X.index.values.tolist(), #transpose=True,
                type = 'heatmap', zmax=1.0, zmin=-1.0,
                xgap=1.5, ygap=1.5, 
                colorscale = [[0, 'rgb(0,0,255)'], [0.5, 'rgb(255,255,255)'], [1, 'rgb(255,0,0)']]#'RdBu',
            )
    data = [trace]
    fig = go.Figure(data = data)
    fig.update_layout(
        margin=dict(
            l=0,
            r=60,
            b=0,
            t=0,
            pad=4
        ),
        #height=380,
        xaxis= {
            #'visible': False
            #'ticks' : "outside",
            'showticklabels': False,
        },
        yaxis= {
            #'visible': False
            #'ticks' : "outside",
            #'showticklabels': False,
        }
    )
    fig.update_traces(colorbar=dict(thickness=15))
    fig.update_layout(
        modebar_remove=['zoom', 'pan', 'select', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale', 'displaylogo']
    )
    return fig



def _plot_heatmap_overview(X):
    trace = go.Heatmap(z=X, x=X.columns.values.tolist(), 
                y=X.columns.values.tolist(), #transpose=True,
                type = 'heatmap', zmax=1.0, zmin=-1.0,
                xgap=1.5, ygap=1.5, 
                colorscale = [[0, 'rgb(0,0,255)'], [0.5, 'rgb(255,255,255)'], [1, 'rgb(255,0,0)']],#'RdBu',
                #colorbar=dict(orientation='h')
            )
    data = [trace]
    fig = go.Figure(data = data)
    fig.update_layout(
        margin=dict(
            l=0,
            r=60,
            b=0,
            t=0,
            pad=4
        ),
        #height=380,
        xaxis= {
            #'visible': False
            #'ticks' : "outside",
            'showticklabels': False,
        },
        yaxis= {
            #'visible': False
            #'ticks' : "outside",
            'showticklabels': False,
        }
    )
    fig.update_traces(colorbar=dict(thickness=15), #colorbar_orientation="h"
    )
    #fig.update_traces(colorbar_orientation='h', selector=dict(type='heatmap'))
    fig.update_layout(
        modebar_remove=['zoom', 'pan', 'select', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale', 'displaylogo']
    )
    return fig

def __plot_heatmap_overview(X):
    fig = px.imshow(X, aspect="auto", 
            zmax=1.0, zmin=-1.0, color_continuous_scale='balance')
    fig.update_layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=4
        ),
        #height=380,
        xaxis= {
            #'visible': False
            #'ticks' : "outside",
            'showticklabels': False,
        },
        yaxis= {
            #'visible': False
            #'ticks' : "outside",
            'showticklabels': False,
        }
    )
    return fig

def _plot_outcome_heatmap_overview(X, ld):
    # create dataframe from preprocessed data and binarized-outcomes
    plot_df = pd.concat([X, pd.get_dummies(ld.ml_target_df["Outcome"])], axis=1)
    _targets = ld.ml_target_df["Outcome"].unique()
    
    plot_df = plot_df.corr()[ld.ml_ordering].T.drop(_targets, axis=1)
    cols = plot_df.columns.values.tolist()
    _newcols = []
    for col in cols:
        if "in Serum or Plasma" in col:
            col = col.replace("in Serum or Plasma", "")
            col += "*"
        elif "in Serum, Plasma or Blood" in col:
            col = col.replace("in Serum, Plasma or Blood", "")
            col += "**"
        elif "Leukocytes" in col:
            col = "Leukocytes [#/volume]***"
        _newcols.append(col)
    trace = go.Heatmap(z=plot_df, x=_newcols, 
                y=plot_df.index.values.tolist(), #transpose=True,
                type = 'heatmap', zmax=1.0, zmin=-1.0,
                xgap=1.5, ygap=1.5, showscale=False,
                colorscale = [[0, 'rgb(0,0,255)'], [0.5, 'rgb(255,255,255)'], [1, 'rgb(255,0,0)']]#'RdBu',
            )
    data = [trace]
    fig = go.Figure(data = data)
    fig.update_layout(
        margin=dict(
            l=100,
            r=0,
            b=30,
            t=0,
            pad=4
        ),
        height=460,
        xaxis= {
            #'dtick': 1,
            #'visible': False
            #'ticks' : "outside",
            #'showticklabels': False,
        },
        yaxis= {
            'dtick': 1,
            #'visible': False
            #'ticks' : "outside",
            #'showticklabels': False,
        }
    )
    fig.update_traces(colorbar=dict(thickness=15))
    fig.update_layout(
        modebar_remove=['zoom', 'pan', 'select', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale', 'displaylogo']
    )
    #st.dataframe(plot_df)
    return fig

def cluster_comparison_barchart(data, show_x_labels=True):
    cbar, cmap_match = _define_colormaps(n_clusters=data.shape[0])
    _data = data.copy()
    #_data = _data.reset_index()
    fig = px.bar(_data.T,color_discrete_sequence=cbar[:data.shape[0]], )#color="Cluster")
    fig.update_layout(
        margin=dict(
            l=10,
            r=0,
            b=30,
            t=0,
            pad=4
        ),
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y= (-1 if show_x_labels else -0.09),
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0.7)",
        ),
        xaxis= {
            "title": "",
            #'dtick': 1,
            'visible': show_x_labels,
            #'ticks' : "outside",
            #'showticklabels': False,
        },
        yaxis= {
            "title": "",
            #'dtick': 1,
            #'visible': False,
            #'ticks' : "outside",
            'showticklabels': False,
        }
    )
    return fig

def overview(ld, df_pp):
    row1_col1, row1_col2, row1_col3, row1_col4 = st.columns([1, 1, 1, 1])
    with row1_col1:
        st.write("")
        st.subheader("Overview - Explore dataset", anchor="Overview")
    with row1_col4:
        _overview_selection = st.selectbox("Choose Visualization Method:", ["Scatter", "Correlation Heat-Map"],
                help=""" Choose from different interactive Visualization types:  
                - __Scatter__: Let's you choose from features to generate a simple 2D scatterplot  
                - __Raw__: Shows the raw DataFrame in tabular form.  
                - __Heat-Map__: Shows feature correlation matrices for scaled features against one another and for all possible Outcomes.  
                """)

    if _overview_selection == "Scatter":
        row2_col1, row2_col2, row2_col3 = st.columns([1, 5, 3])
        with row2_col1:
            opts = sorted(ld.raw_ehd.columns.values.tolist())
            scatter_x_select = st.selectbox("X-axis", opts, index=21,
                                key="scatter-x")
            scatter_y_select = st.selectbox("Y-axis", opts, index=len(opts)-19,
                                key="scatter-y")

        with row2_col2:
            st.plotly_chart(_plot_scatter_overview(ld.raw_ehd, scatter_x_select, scatter_y_select, ld.ml_target_df["Outcome"]),
                use_container_width=True, config={'displayModeBar': False})
        with row2_col3:
            st.write("Raw Data")
            st.dataframe(ld.raw_ehd.set_index("to_patient_id"),
            # height=300
            )

    elif _overview_selection == "Raw":
        row2_col1, row2_col2 = st.columns([4, 3])
        with row2_col1:
            st.dataframe(ld.raw_ehd.set_index("to_patient_id"), height=300)

        with row2_col2:
            if _overview_selection == "Raw":
                st.markdown(f"Dataset shape: {ld.raw_ehd.shape}")

                st.dataframe(pd.DataFrame(ld.raw_ehd.isna().sum(),
                        columns=["Sum of NaNs"]).sort_values(by="Sum of NaNs", ascending=False),
                        height=260)
    elif _overview_selection == "Correlation Heat-Map":
        row2_col1, row2_col2, row2_col3 = st.columns([2.5, 0.5, 3])
        with row2_col1:
            st.plotly_chart(_plot_heatmap_overview(df_pp.corr()))
        with row2_col3:
            #st.write(df_pp.shape)
            #foo = pd.concat([df_pp, pd.get_dummies(ld.ml_target_df["Outcome"])], axis=1)
            #st.dataframe(df_pp)
            #st.dataframe(foo.corr()[ld.ml_target_df["Outcome"].unique()])
            st.plotly_chart(_plot_outcome_heatmap_overview(df_pp, ld), use_container_width=True,
                 config={'displayModeBar': False})


def overview_draft(ld:LungData, model, device, df_pp, rdf, rdfs_targets):

    ### PREPARE SOME STUFF
    _boolean = ld.pre_known_features_onehot.copy()
    ## remove known non-boolean cols
    _boolean.remove("Body mass index (BMI) [Ratio]")
    _boolean.remove("Oral temperature")
    ## add known non-numeric cols
    _boolean = _boolean + ['Urine.protein_Abnormal', 'Urine.protein_Normal', 'Urine.protein_nan',
                        'Microscopic_hematuria.above2', 'Proteinuria.above80']
    _boolean = sorted(_boolean, key=lambda v: v.lower())
    st.session_state["boolean-columns"] = _boolean

    ### OVERVIEW
    overview(ld, df_pp)


    st.markdown("---")
    st.subheader("Imputation", anchor="Imputation")

    _imputation_options = dict(
        zip(["Nearest-Neighbor", "Mean", "Median", "Missing Indicator", "Iterative (Multiple-Imputation)"],
            ["nn", "mean", "median", "mi", "iterative"]))

    
    row2_col1, row2_placeholder, row2_col2, row2_col3, row2_col4 = st.columns([1,0.15,1,1,1])

    with row2_col1:
        imp_selection = st.selectbox("Imputation", _imputation_options.keys(),
            help="""Imputation method to use. The chosen method is applied to ```Clustering``` below.  
                - __Mean__:   Average of all non-missing values.  
                - __Median__: Median of all non-missing values.  
                - __Missing Indicator__: Fill with indicator. Often ```0``` or ```-1```.  
                - __Nearest-Neighbor__: Impute values based on ```k``` nearest-neighbors. Uniform Distance.  
                - __Iterative__: Multiple-Imputation. Learns multiple regression-predictors from randomly chosen available variables to predict missing values.  

            """)
        imputation = _imputation_options[imp_selection]
        #st.write(f"Chosen {imputation}")

        mi_val = st.number_input("Missing Indicator Fill Value", step=1, value=0, 
                disabled= True if imputation != "mi" else False,
                help="""The value that is used to fill missing values.
                    Popular options are ```0``` or ```-1```.""")

        nn = st.number_input("Nearest Neighbors", step=1, value=5, 
                disabled= True if imputation != "nn" else False,
                help="""How many neighbors are taken into account for imputation. Default ```5```.""")
    
    _st = time.time()
    _ehd = _preproc_ehd(ld, imputation, nn, mi_val)
    _ehd = _ehd.drop(ld._targets, axis=1, errors="ignore")
    
    imputation_settings = f"{imputation} {nn if imputation == 'nn' else ''}{mi_val if imputation == 'mi' else ''}"
    logging.info("Running {} with settings {} took: {}".format("imputation", imputation_settings, round(time.time()-_st, 3)))


    plot_imputation_examples(_ehd, row2_col2, row2_col3, row2_col4, _height=4.)

    _ehd_scaled = pd.DataFrame(StandardScaler().fit_transform(_ehd), columns=_ehd.columns)
    
    #st.dataframe(_ehd_scaled)

    st.markdown("---")
    #st.markdown("#### Clustering")
    #st.subheader("Clustering", anchor="Clustering")
    #build_clustering(ld, _ehd_scaled, imputation_settings)

    targets = ld._targets
    df_verbose_targets = ld.verbose_target_df
    #df_targets = ld.target_df

    if "clustering_results" not in st.session_state:
        st.session_state["clustering_results"] = initialize_cluster_results()
    clustering_results = st.session_state["clustering_results"]
    #st.dataframe(df_targets)

    col1, col_placeholder, col2, col3 = st.columns([1, 0.2, 4, 2])
    with col1:
        st.subheader("Clustering", anchor="Clustering")
        method = st.selectbox("Clustering Method", ["K-means", "Ward",]) #"DBSCAN"])# "DBSCAN"])
        with st.form("clustering-input"):
            if method is not "DBSCAN":
                n_clusters = st.number_input("N Clusters", step=1, value=4,
                    min_value=2, max_value=20)

                seed = st.number_input("seed", step=1, value=42,
                    min_value=0, max_value=9999)
            elif method == "DBSCAN":
                eps = st.number_input("Epsilon", step=0.01, value=0.1, min_value=0.0, max_value=25.0)
            submit = st.form_submit_button("Cluster!")

            if submit:
                st.session_state["clustering-inputs"] = _form_inputs(imputation_settings, method, n_clusters, seed)
                #classes, df_scaled, df, df_targets, targets = _cluster(ld, inputs)
                st.session_state["classes"], st.session_state["clustered_df"] = _cluster(ld, _ehd_scaled, st.session_state["clustering-inputs"])
                # data = create_cluster_means(_ehd, st.session_state["classes"])
                # data_bool = data[st.session_state["boolean-columns"]]
                # legend, indicator, _data = prepare_echart_radar_data(data_bool)
                # st.session_state["cluster-means"] = data
                print("CLSUTERE HERE")

        tsne = st.checkbox("Use t-SNE representation", value=True)
    
    # if not yet submitted, run it once
    st.session_state["clustering-inputs"] = _form_inputs(imputation_settings, method, n_clusters, seed)
    st.session_state["classes"], st.session_state["clustered_df"] = _cluster(ld, _ehd_scaled, st.session_state["clustering-inputs"])

    # with r1_col1:
    #     t = st.selectbox("Target", targets)
    t = "last.status"
    with col2:
        #st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)

        #st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:normal;padding-left:2px;}</style>', unsafe_allow_html=True)

        choose = st.selectbox("",("Scatter", "Cluster Characteristics"))
        if choose == "Scatter":
            if tsne:
                data = convert_to_tsne(_ehd_scaled)
                #st_echarts(echart_scatterplot(data, st.session_state["classes"], 
                #                    n_clusters))
                fig = _scatterplot_plotly(data, st.session_state["classes"], 
                                        n_clusters)
            else:
                #st_echarts(echart_scatterplot(_ehd_scaled, st.session_state["classes"], 
                #                    n_clusters))
                fig = _scatterplot_plotly(st.session_state["clustered_df"], st.session_state["classes"],
                       n_clusters)
            st.plotly_chart(fig, use_container_width=True, 
                config={'displayModeBar': False},
            )
        elif choose == "Cluster Characteristics":
            boolean_only = st.checkbox("Boolean variables only", value=True, help="""
            Toggle between bolean and numerical variables. 
            """)
            if boolean_only:
                # #if "cluster-means" not in st.session_state:
                data = create_cluster_means(_ehd, st.session_state["classes"])
                data_bool = data[st.session_state["boolean-columns"]]
                fig = cluster_comparison_barchart(data_bool)
                st.plotly_chart(fig, use_container_width=True, 
                    config={'displayModeBar': False},
                )
                # legend, indicator, _data = prepare_echart_radar_data(data_bool)
                # #st.session_state["cluster-means"] = data
                # st_echarts(echart_radar(_data, legend, indicator), height="400px")
            else:
                data = create_cluster_means(_ehd_scaled, st.session_state["classes"])
                #topfeats = clustering_get_top_feats(_ehd, st.session_state["classes"], 15)
                #fig = cluster_comparison_barchart(data.drop(st.session_state["boolean-columns"], axis=1), show_x_labels=False)
                fig = cluster_comparison_barchart(data, show_x_labels=False)
                st.plotly_chart(fig, use_container_width=True, 
                    config={'displayModeBar': False},
                )
                # print(data[topfeats])
                # legend, indicator, _data = prepare_echart_radar_data(data[topfeats])
                # st_echarts(echart_radar(_data, legend, indicator), height="400px")
            
    with col3:
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        # TODO: mutate df on the fly is not working for now 
        # the clustering_resutls are saved in the st.session_state and initialized above
        # results are appended and the state is updated afterwards
        #st.write(clustering_results.shape)
        clustering_results = _compute_clustering_metrics(clustering_results,
                        st.session_state["clustered_df"], st.session_state["classes"],
                        st.session_state["clustering-inputs"])
        update_clustering_results(clustering_results)
        
        ## two plots
        fig = _barplot_plotly(clustering_results.drop_duplicates(subset="Settings", keep="last"), "Silhouette")
        st.plotly_chart(fig, use_container_width=True)
        fig = _barplot_plotly(clustering_results.drop_duplicates(subset="Settings", keep="last"), "Calinski-Harabasz")
        st.plotly_chart(fig, use_container_width=True)

    
    _prettycols = ['Hospitalized (only)',
        'ICU', 'Ventilated', 'ICU + Ventilated',
        'Deceased', 'Deceased + Ventilated',
        'Deceased + ICU + \nVentilated']



    st.markdown("---")
    st.subheader("Segmentation", anchor="Segmentation")
    
    set_constants()
    st.session_state["radiomics-record-prediction"] = upload(model, device, disable_preproc_options=False, return_preds=True)

    #st.write(st.session_state["radiomics-record-prediction"])

    #st.markdown("__Radiomics Prediction__:")
    c1,c2 = st.columns([0.5,2])
    with c1:
        if "radiomics-record-prediction" in st.session_state:
            #st.write(st.session_state["radiomics-record-prediction"])
            st.write("")
        else: 
            st.write("Upload a picture!")
    with c2:
        preds_chart_placeholder = st.empty()

    data = st.session_state["radiomics-record-prediction"] if st.session_state["radiomics-record-prediction"] is not None else dummies(_prettycols)
    #if "radiomics-record-prediction" in st.session_state:
    with preds_chart_placeholder:
        st_echarts(plot_preds_bar_echart(data), height="300px", key="BarResults")
        #st.write(st.session_state["radiomics-record-prediction"])

    st.markdown("---")
    st.subheader("Radiomics Features", anchor="Radiomics")
    col1, col2 = st.columns([1,3])
    with col1:
        st.markdown("""[Radiomics](https://pyradiomics.readthedocs.io/en/latest/features.html) is used to extract texture and shape features of
        various forms and sizes from the image and the segmented area, so a valid segmentation is of importance to pre-select
        the areas that are used by radiomics. The best scenario is that some radiologist segments those images but this is not done here. We use 
        strategies for segmentation powered by transfer learning (you can try different pre-processing techniques and models above).  
        """)
        st.markdown("""
        """)
        rad_viewtype = st.radio(
            "Data View: ",
            ('Condensed', 'Full'))
        sort_by = st.selectbox("Sort by", ld.ml_target_df["Outcome"].unique())

    with col2:
        if rad_viewtype == "Condensed":
            targets = ld.ml_target_df["Outcome"].unique()
            _drops = ld._targets + list(ld.ml_target_df["Outcome"].unique())
            plot_df = rdfs_targets.corr()[targets].sort_values(by=sort_by).drop(_drops)[ld.ml_ordering]
            #st.dataframe(plot_df.T)
            st.plotly_chart(_plot_heatmap_targets(plot_df.T), use_container_width=True)
        elif rad_viewtype == "Full":
            st.plotly_chart(_plot_heatmap_overview(rdf.corr()),
                #config={'displayModeBar': False}
            )
    
    st.markdown("---")
    st.subheader("Prediction", anchor="Prediction")
    col1, col2 = st.columns([1,3])
    with col1:
        models = [SVC(probability=True), RandomForestClassifier(), LR(), XGBClassifier(), MLPClassifier()]
        model_options = ["SVC", "Random Forest", "Logistic Regression", "XGBoost", "MLP"]
        dataset = st.selectbox("Choose Dataset: ", ["EHD", "Radiomics", "Merged"])
        pred_model = st.selectbox("Choose Model: ", model_options )
        model_inputs = st.container()
    with col2:
        st.subheader("Results + Confusion Matrix")
        #st.dataframe(_ehd)
        pred_results_overview = st.container()

    with model_inputs:
        _models = {k:v for k,v in zip(model_options, models)}
        with st.form("prediction-inputs"):
            if pred_model == "SVC":
                c = st.number_input("C", min_value=0.0, step=0.01, value=1.0)
                solver = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
                settings = {"C": c, "kernel": solver}
            elif pred_model == "Random Forest":
                n_trees = st.number_input("Number of trees", min_value=10, max_value=1000, value=100)
                max_depth = st.number_input("Max depth", min_value=0, max_value=1000, value=0,
                 help="Max depth 0 translates to unlimited depth.")
                settings = {"n_estimators": n_trees,
                    "max_depth": (None if max_depth==0 else max_depth)}
            elif pred_model == "Logistic Regression":
                c = st.number_input("C", min_value=0.0, step=0.01, value=1.0)
                max_iter = st.number_input("Maximum iterations", min_value=100, max_value=1000, value=100)
                solver = st.selectbox("solver", [ "lbfgs", "newton-cg", "liblinear", "saga"])
                settings = {"max_iter": max_iter, "solver": solver}
            elif pred_model == "XGBoost":
                n_estimators = st.number_input("Number of trees", min_value=10, max_value=1000, value=100)
                learning_rate = st.number_input("Learning rate (eta)", min_value=0.0, step=0.01, value=0.01)
                reg_lambda = st.number_input("Regularization term (lambda)", min_value=0.0, step=0.001, value=1e-05)
                complexity_gamma = st.number_input("Complexity (gamma)", min_value=0.0, step=0.001, value=1e-05)
                settings = {"n_estimators": n_estimators, "learning_rate": learning_rate, 
                            "reg_lambda": reg_lambda, "gamma": complexity_gamma}

            elif pred_model == "MLP":
                hidden_layers = st.number_input("Hidden Layers", min_value=10, max_value=1000, value=100)
                activation = st.selectbox("Activation", ['relu', 'logistic', 'tanh',])
                learning_rate = st.number_input("Learning rate", min_value=0.0, step=0.01, value=0.01)
                solver = st.selectbox("Solver", ['adam', 'lbfgs', 'sgd'])
                settings = {"hidden_layer_sizes": (hidden_layers,), "activation": activation, "learning_rate_init": learning_rate, "solver": solver}

            seed = st.number_input("Random Seed", min_value=0, value=42)
            balanced_weights = st.checkbox("Use balanced weights", value=True)
            submit = st.form_submit_button("Train and Predict")
            if submit:
                _st = time.time()
                if pred_model in ["Random Forest", "Logistic Regression", "SVC"] and balanced_weights:
                    settings["class_weight"] = "balanced"
                settings["random_state"] = seed
                predmodel = _models[pred_model]
                predmodel.set_params(**settings)
                print(predmodel)
                res, cm = predict(predmodel, _ehd, ld.ml_target_df["y"], settings, seed)
                if "manual-prediction" not in st.session_state:
                    st.session_state["manual-prediction"] = pd.DataFrame()
                st.session_state["cm"] = cm
                prev_res = st.session_state["manual-prediction"]
                new_res = prev_res.append(res).reset_index(drop=True)\
                    .drop_duplicates(subset=["Settings","Metric"], keep="last")

                st.session_state["manual-prediction"] = new_res
                logging.info("Train and predict with {} took {}s".format(predmodel, round(time.time()-_st, 3)))

    #st.markdown("---")
    #st.subheader("Merge Feats", anchor="Merged")
    with pred_results_overview:
        if "manual-prediction" in st.session_state:
            preds = st.session_state["manual-prediction"]
            
            fig = predictions_lineplot(preds)
            st.plotly_chart(fig, use_container_width=True,
                config={'displayModeBar': False}
            )

            cm = st.session_state["cm"]
            st.markdown(cm.to_markdown())
        else:
            st.write("Train classifiers to compare results")

def predictions_lineplot(data):
    fig = px.line(data, x="Settings", y="Score", color="Metric", markers=True, 
        hover_data={ "Settings":False,
            "Metric":True, 
            "Score":True})
    fig.update_layout(
        margin=dict(
            l=10,
            r=0,
            b=30,
            t=0,
            pad=4
        ),
        height=300,
        # legend=dict(
        #     orientation="h",
        #     yanchor="bottom",
        #     y= -0.09),
        #     xanchor="left",
        #     x=0.0,
        #     bgcolor="rgba(255,255,255,0.7)",
        # ),
        xaxis= {
            "title": "Setting",
            #'dtick': 1,
            #'visible': False,
            #'ticks' : "outside",
            'showticklabels': False,
        },
        yaxis= {
            "title": "Score",
            "range": [0,1]
            #'dtick': 1,
            #'visible': False,
            #'ticks' : "outside",
            #'showticklabels': False,
        },
    )
    fig.update_traces(line=dict(width=2), marker=dict(size=6))
    #fig.update_xaxes(showspikes=True)
    #fig.update_yaxes(showspikes=True, spikecolor="#444")
    fig.update_layout(hovermode="x")
    #fig.update_layout(hovermode="x unified")
    return fig


def predict(model, ehd, _y, settings, seed=42):
    scoring = { 'accuracy': make_scorer(accuracy_score),
            'balanced_accuracy': make_scorer(balanced_accuracy_score),
            'roc_auc': make_scorer(roc_auc_score, multi_class="ovo", needs_proba=True, average="weighted"),
            'recall': make_scorer(recall_score, average = 'micro'),
        }
    _targets = ['Hospitalized',
        'Vent', 'ICU', 'ICU + Vent.',
        'Deceased', 'Deceased + ICU',
        'Deceased + ICU + Vent.']

    y = _y.copy()
    y[y==5] = 7
    #print(y.value_counts())
    X_train, X_val, y_train, y_val = train_test_split(ehd, y, test_size=0.25, random_state=seed,
                                            stratify=y
                                            )

    pipe = Pipeline([("scaler", StandardScaler()), 
                    #("imputer", KNNImputer(n_neighbors=5)),
                    #("scaler2", StandardScaler()), 
                    ("clf", model)
                    ])

    pipe.fit(X_train, y_train)
    y_hat = pipe.predict(X_val)
    #print(pd.DataFrame(y_hat).value_counts())

    settings["Classifier"] = type(model).__name__
    settings_str = str(settings)
    results = pd.DataFrame()

    res = {}
    res["Settings"] = [settings_str, settings_str, settings_str, settings_str, settings_str]
    res["Score"] = [f1_score(y_val, y_hat, average="weighted"), accuracy_score(y_val, y_hat),
        balanced_accuracy_score(y_val, y_hat), recall_score(y_val, y_hat, average="weighted"), 
        roc_auc_score(y_val, pipe.predict_proba(X_val), multi_class="ovo", average="weighted")]
    res["Metric"] = ["F1", "Accuracy", "Balanced Accuracy", "Recall", "ROC-AUC"]
    
    results = pd.DataFrame.from_dict(res, orient="index").T

    cm = confusion_matrix(y_val, y_hat)
    cm_pretty = pd.DataFrame(cm, index=_targets, columns=_targets)
    #print(cm)

    #print(results)
    return results, cm_pretty