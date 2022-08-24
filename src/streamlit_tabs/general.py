import os, sys, logging, time
from io import BytesIO
from typing import List
from sklearn.metrics import label_ranking_loss

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.utils import resample
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import ../db.py
from clustering import compute_cluster_metrics, cluster, plot_cluster_barplot, compute_target_statistics_per_cluster
from ehd_classification.utils import compute_bmi, create_patient

import utils

from PIL import Image
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from streamlit_echarts import st_echarts, st_pyecharts
from pyecharts import options as opts
from pyecharts.charts import Bar, Pie

pretty_labels = utils.get_pretty_labels()

#pd.options.display.float_format = '{:, .2f}'.format

import joblib

#@st.experimental_memo
def prepare_constants(ehd, clusters):
    _st = time.time()
    model = joblib.load("models/ehd/svm_roc_auc_trained_on_all.pkl")
    st.session_state["model"] = model

    st.session_state["kn_class"] = KNeighborsClassifier(n_neighbors=5)
    st.session_state["scaler"] = StandardScaler()

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
    cluster_means = _ehd.groupby("cluster").mean()

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

    logging.info("Running {} took: {}".format("prepare_constants", round(time.time()-_st, 3)))
    return cluster_means


@st.experimental_memo
def _form_patient_inputs(age, gender, height, weight, smoking_status, heart_failure, 
                        diabetes, high_blood_pressure, kidney_replacement, kidney_transplant,
                        malignancies, copd, cluster):
    _ages = {"18 - 59": "[18,59]",
            "59 - 74": "[59,74]",
            "74 - 90": "[74,90]"}
    inputs = {"age": _ages[age], "gender":gender, "BMI": compute_bmi(weight, height),
            "smoking_status": smoking_status, "heart_failure": heart_failure,
            "dm_v": int(diabetes), "htn_v": int(high_blood_pressure),
            "kidney_replacement_therapy": int(kidney_replacement),
            "kidney_transplant": int(kidney_transplant),
            "malignancies_v": int(malignancies), "copd_v": int(copd),
            "cluster": cluster
    }
    verbose_inputs = {"Age": age, "gender":gender, "BMI": compute_bmi(weight,height),
                    "Smoking Status": smoking_status, "Heart Failure": heart_failure,
                    "Diabetes": diabetes, "High blood pressure": high_blood_pressure, 
                    "Kidney Replacement Therapy": kidney_replacement, 
                    "Kidney transplant": kidney_transplant, "Malignancies": malignancies,
                    "COPD": copd, "Cluster": cluster}
    return inputs, verbose_inputs


@st.cache
def create_patient_record(inputs:dict, columns, cluster_means):
    """
    Create patient record from archetype and input.
    """
    patient = create_patient(inputs, columns)
    #print(patient.iloc[0, 15:20])

    #notnas = patient.notna().values[0]
    #input_columns = patient.columns.to_numpy()[notnas]
    #print(input_columns)

    # if "kn_class" not in st.session_state:
    #     print("kn_class not found - rerunning...")
    #     st.experimental_rerun()
    # else:
    knc = st.session_state["kn_class"]

    sclr = st.session_state["scaler"]
    infered_cluster = knc.predict(sclr.transform(patient[st.session_state["input-columns"]]))
    neighbors = knc.kneighbors(sclr.transform(patient[st.session_state["input-columns"]]), n_neighbors=5)[1][0]
    print(f"PREDICTION: {infered_cluster} --- {neighbors}")
    st.session_state["patient-infered-cluster"] = infered_cluster
    st.session_state["patient-infered-neighbors"] = neighbors

    # if its Automatic we just use the pre-trained pipeline with scaler and imputer in it
    # if not we impute prior with cluster data
    if inputs["cluster"] != "Automatic":
        try: 
            cluster_choice = int(inputs["cluster"])
        except Exception as e:
            raise e
        cluster_data = cluster_means.iloc[cluster_choice - 1] #-1 because we have rows from 0-3, cluster 1-4
        patient = patient.fillna(cluster_data)
    #print(patient.iloc[0, 15:20])
    return patient
    

def jitter_dots(dots):
    offsets = dots.get_offsets()
    jittered_offsets = offsets
    # only jitter in the x-direction
    jittered_offsets[:, 0] += np.random.uniform(-0.3, 0.3, offsets.shape[0])
    dots.set_offsets(jittered_offsets)

def rand_jitter(arr, strength=.01):
    stdev = strength * (max(arr) - min(arr))
    np.random.seed(42)
    return arr + np.random.randn(len(arr)) * stdev

def getImage(path, size=0.01):
   return OffsetImage(plt.imread(path, format="png"), zoom=size)

def jitter(x, y, s=20, c='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None, **kwargs):
    return scatter(rand_jitter(x), rand_jitter(y), s=s, c=c, marker=marker, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths, **kwargs)

def jitter_hist(data, xpos=0, width=1.8):
    counts, edges = np.histogram(data, bins=40)

    centres = (edges[:-1] + edges[1:]) / 2.
    yvals = centres.repeat(counts)

    max_offset = width / counts.max()
    offsets = np.hstack((np.arange(cc) - 0.5 * (cc - 1)) for cc in counts)
    
    xvals = xpos + (offsets * max_offset)

    return xvals, yvals

@st.experimental_memo
def icons_overview(input_df):
    resampled_df = input_df.copy()
    resampled_df["Hospitalization Outcome"].replace({"Deceased": 0,
                                            "Ventilated": 1,
                                            "Intensive Care Unit": 2},
                                             inplace=True)
                                            
    #st.dataframe(resampled_df)
    fig, ax = plt.subplots(1, 4, sharey=True, figsize=(12,4))
    for i, _cluster in enumerate(range(1,5)):
        #print(i, _cluster)
        _thiscluster = resampled_df[resampled_df.Cluster == _cluster]

        _x = rand_jitter(_thiscluster["Hospitalization Outcome"], strength=0.06)
        _y = _thiscluster["Days Hospitalized"]
        #_x, _y = jitter_hist(_thiscluster["Days Hospitalized"], xpos=_thiscluster["Hospitalization Outcome"]) # hist jitter
        _icons = _thiscluster["icon"]
        #ax[i].scatter(_x, _y, s=20)
        for x0, y0, icon in zip(_x, _y, _icons):
            ab = AnnotationBbox(getImage(f"images/icons/{icon}", size=0.05), (x0, y0), frameon=False)
            ax[i].add_artist(ab)
        #ax[i].scatter(dots)
        #dots
        #ax[i].add_collection(dots)
        
        #sns.swarmplot(data = _thiscluster, x="Hospitalization Outcome", y="Days Hospitalized",
        #                ax=ax[i])
        ax[i].set_title(f"Cluster {_cluster}")
        if i == 0:
            ax[i].set_ylabel("Days Hospitalized")
        else:
            ax[i].set_ylabel("")
        ax[i].set_xticks([0, 1, 2])
        ax[i].set_xticklabels(['Decased', 'Ventilated', 'Intensive Care Unit\nAdmission'],
                            #fontsize='large'
                            )
        ax[i].tick_params("x", rotation=360-45)
        ax[i].set_xlim((-0.5, 2.5))
        ax[i].set_ylim((0, 100))
        ax[i].grid(False)
        #ax[i].axis('off') removes EVERYTHING
        for side in ["top", "right", "bottom", "left"]:
            ax[i].spines[side].set_visible(False)
        
    fig.tight_layout()
    return fig

@st.experimental_memo
def overview_plotly(resampled_df):
    fig = px.strip(resampled_df, x="Hospitalization Outcome", y="Days Hospitalized",
                facet_col="Group", color="Hospitalization Outcome", 
                #labels={0:"Cluster 1"}, 
                #facet_col_spacing=0.2,
                #range_y=(0,1)
                )
    #fig = px.violin(_ehd, x="Outcome", y="Days Hospitalized", facet_col="Cluster", color="Outcome",
    #        box=False, points="all")
    fig.update_layout(showlegend=False)
    fig.for_each_trace(lambda t: t.update({"marker":{"size":11, "opacity":0.8, "symbol":"hexagon",
                                            "line":{"width":2, "color":"DarkSlateGrey"}
                                            }}
                                        ))
    #fig.update_traces(marker=dict(size=18,
    #                          line=dict(width=2,
    #                                    color='DarkSlateGrey')),
    #              selector=dict(mode='markers'))
    fig.update_layout(
        plot_bgcolor="#f8f8f8",
        margin=dict(
            l=0,
            r=00,
            b=00,
            t=20,
            pad=4
        ),
        # legend=dict(
        #     orientation="h",
        #     yanchor="top",
        #     y=1.,
        #     xanchor="left",
        #     x=0.01,
        #     bgcolor="rgba(255,255,255,0.7)",
        # ),
        xaxis= {
            'dtick': 1,
            #'range': [0.2, 1],
            #'showgrid': True, # thin lines in the background
            #'zeroline': False, # thick line at x=0
            #'visible': False,  # numbers below
        }, 
        yaxis= {
            #'range': [0.2, 1],
            #'showgrid': True, # thin lines in the background
            #'zeroline': False, # thick line at x=0
            #'visible': False,  # numbers below
        },
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("=", " ")))
    return fig

@st.experimental_memo
def simple_scatter(resampled_df, figsize=(12,4)):
    count_df = resampled_df.groupby(["Group", "Hospitalization Outcome"]).count().iloc[:,0].reset_index().rename({"kidney_replacement_therapy":"count"}, axis=1)
    #st.dataframe(count_df)

    _outcomes = ["Intensive Care Unit", "Ventilated", "Deceased"]
    _ids = [1,2,3]
    _clusters = [i for i in range(1,5)]

    arrays = np.full((4,300), 0)
    for _c in _clusters:
        _counter = 0
        for id, _out in zip(_ids, _outcomes):
            count = count_df[count_df["Hospitalization Outcome"]==_out].query("Group == @_c")["count"].values[0]
            arrays[_c-1, _counter:_counter+count] = id # count from [1-4] -> [0,3]
            _counter += count
    
    fig, ax = plt.subplots(1,4, sharex=True, sharey=True, figsize=figsize)

    coords_x = []
    coords_y = []
    for _y in range(20,0,-1):
        for _x in range(0,20):
            coords_x.append(_x)
            coords_y.append(_y*0.5)

    palette = {0: "white", 1:"#fc8d62", 2:"#8da0cb", 3:"#e78ac3", 4:"black"}
    clist = list(palette.values())
    for _i, _c in enumerate(_clusters):
        #sns.set(rc={'axes.facecolor':'cornflowerblue', 'figure.facecolor':'cornflowerblue'})
        _df = pd.DataFrame([coords_x, coords_y, arrays[_i]]).T
        print(_df.shape, _df.head())
        _df = _df.dropna(0)
        sns.scatterplot(data=_df, x=0, y=1, hue=2, markers="8", palette=palette, s=55,
                 ax=ax[_i], edgecolor="DarkSlateGrey", alpha=0.9)
        sns.despine(right=False, bottom=False, left=True, ax=ax[_i])
        ax[_i].set(ylim=(2.6,15))
        ax[_i].set_xticks([])
        ax[_i].set_yticks([])
        ax[_i].set_xlabel("")
        ax[_i].set_ylabel("")
        if _i != 0:
            ax[_i].get_legend().remove()
        else:
            msize = 9
            line0 = mlines.Line2D(range(1), range(1), color="white", marker='o',
             markersize=msize, markerfacecolor=clist[0], markeredgecolor="DarkSlateGrey")
            line1 = mlines.Line2D(range(1), range(1), color="white", marker='o', markersize=msize, markerfacecolor=clist[1], markeredgecolor="DarkSlateGrey")
            line2 = mlines.Line2D(range(1), range(1), color="white", marker='o', markersize=msize, markerfacecolor=clist[2], markeredgecolor="DarkSlateGrey")
            line3 = mlines.Line2D(range(1), range(1), color="white", marker='o', markersize=msize, markerfacecolor=clist[3], markeredgecolor="DarkSlateGrey")
            ax[_i].legend((line0, line1,line2,line3),('Hospitalised','ICU','Ventilated', 'Deceased'),
                numpoints=1, loc=3, frameon=True, ncol=1, fontsize=8)
            #ax[_i].legend(handles=[red_patch])
            #ax[_i].legend(handles=[red_patch])
            #ax[_i].legend([0,1],["s", "d"])
    #buf = BytesIO()
    #fig.savefig(buf, format="png")
    return fig

def build_prediction(n_clusters=4):
    with st.form("prediction-inputs"):
        ## row1
        col1, col2, col3, col4, col5, col6, col7 = st.columns([1.5,1.5,1.5,1.5,1.5,1.5,1.5])
        with col1:
            #age = st.number_input("Pick patients age", step=1, 
            #min_value=18, max_value=99, value=35)
            age = st.selectbox('Select Age', ('18 - 59', '59 - 74', '74 - 90'))
            diabetes = st.checkbox("Diabetes")
        
        with col2:
            gender = st.selectbox('Select gender', ('Male', 'Female'))
            htn = st.checkbox("High Blood Pressure")

        with col3:
            height = st.number_input("Patients height (cm)", step=1, 
                                    min_value=100, max_value=220, value=185)
            kidney_therapy = st.checkbox("Kidney replacement therapy")
        with col4: 
            weight = st.number_input("Patients weight (kg)", step=1, 
                                    min_value=35, max_value=250, value=85)
            kidney_transplant = st.checkbox("Kidney transplant")
        with col5:
            smoking_status = st.selectbox('Smoking Status', ('Never',
                'Former', 'Current', 'No Answer'))
            malignancies = st.checkbox("Malignancies")
        with col6:
            hfef_v = st.selectbox("Heart Failure", options=["No", "CAD", "HFpEF", "HFrEF"], help="""- __CAD__: _Coronary Artery Disease_
- __HFpEF Heart failure with preserved ejection fraction__: Heart failure is diagnosed but the pumping is still valid.  
- __HFrEF Heart failure with reduced ejection fraction__: Hearts pumping function is reduced.""")

            copd_v = st.checkbox("Chronic Obstructive Pulmonary Disease",
                    help="Chronic, progressive lung disease characterized by long-term respiratory symptoms and airflow limitation.")

        with col7: 
            cluster = st.selectbox('Select Group',
                ["Automatic"] + [i for i in range(1, n_clusters+1)])
            st.write(" ")
            st.write(" ")
            st.write(" ")
            submit = st.form_submit_button("Predict")
            if submit:
                inputs, verbose_inputs = _form_patient_inputs(age, gender, height, weight,
                             smoking_status, hfef_v, diabetes, htn, kidney_therapy,
                             kidney_transplant, malignancies, copd_v, cluster)
                
                st.session_state["user_inputs"] = inputs
                st.session_state["cluster_method"] = inputs["cluster"]
                st.session_state["user_inputs_verbose"] = verbose_inputs


#@st.experimental_memo(suppress_st_warning=True)
def display_cluster_help():
    cols = st.columns([0.1, 1,1,1,1])
    with cols[1]:
        st.markdown("""__Group 1 - Healthy Young to Middle Aged Patients__:   
This group of patients was unlikely to have serious medical preconditions.
Patients in that group did have the lowest probability of being Ventilated,
admissioned to the Intensive Care Unit (ICU) or mortal outcomes during their hospitalization stay.
There have been more Female patients in this particular Group.
        """)
        #st.markdown("""Out of a __100__ Patients in this group, __5__ patients were admissioned to the __ICU__, __4__ had to be __ventilated__ and __1 died__.""")
    
    with cols[2]:
        st.markdown("""__Group 2 - Less Healthy Young to Middle aged__:   
This group of patients is consisting of young to middle aged, predominantly male patients with more serious decease progressions. 
Patients were more likely to have preconditions like _diabetes_ and were on average reporting higher _Body Mass Index_ than those of Cluster 1. 
More Male Patients were in this risk Group.
        """)
        #st.markdown("""Out of a __100__ Patients in this group, __27__ patients were admissioned to the __ICU__, __24__ had to be __ventilated__ and __8 died__.""")
    
    with cols[3]:
        st.markdown("""__Group 3 - Elderly High Risk Patients__:   
Patients in this group were oldest on average. Many had medical preconditions like _diabetes_, _heart issues_ or _Malignancies_ in companion to being past smokers.
In comparison to the other groups, they have the largest amount of patients dying without receiving special medical therapy while reporting the second lowest hospitalization durations.
Patients in this group had a fast and serious, often deadly, disease progression.""")
    #st.markdown("""Out of a __100__ Patients in this group, __18__ patients were admissioned to the __ICU__, __12__ had to be __ventilated__ and __25 died__.""")

    with cols[4]:
        st.markdown("""__Group 4 - High Risk__:   
Patients in this group were present from all age groups. They include patients with the most serious decease progressions. 
Patients were characterized by suffering numerous medical preconditions including _kidney replacement therapy_, _high blood pressure_ and _malignancies_. 
        """)
        #st.markdown("""Out of a __100__ Patients in this group, __55__ patients were admissioned to the __ICU__, __46__ had to be __ventilated__ and __49 died__.""")

def display_cluster_per100():
    cols = st.columns([0.15, 1,1,1,1])
    # ICU 1:"#fc8d62", VENT 2:"#8da0cb", DECEASED 3:"#e78ac3",
    with cols[1]:
        #st.markdown("""Out of a __100__ Patients in this group, __5__ patients were admissioned to the __ICU__, __4__ had to be __ventilated__ and __1 died__.""")
        t = """<p>
                Out of a <span style="font-weight:bold;">100</span> Patients in this group, 
                <span style="font-weight:bold;">5</span> patients were admissioned to the 
                <span style="font-weight:bold; color:#fc8d62">ICU</span>, 
                <span style="font-weight:bold;">4</span> patients had to be 
                <span style="font-weight:bold; color:#8da0cb">Ventilated</span>, and 
                <span style="font-weight:bold;">1</span> 
                <span style="font-weight:bold; color:#e78ac3">died</span>.
            </p>"""
        st.markdown(t, unsafe_allow_html=True)

    with cols[2]:
        #st.markdown("""Out of a __100__ Patients in this group, __27__ patients were admissioned to the __ICU__, __24__ had to be __ventilated__ and __8 died__.""")
        t = """<p>
                Out of a <span style="font-weight:bold;">100</span> Patients in this group, 
                <span style="font-weight:bold;">27</span> patients were admissioned to the 
                <span style="font-weight:bold; color:#fc8d62">ICU</span>, 
                <span style="font-weight:bold;">24</span> patients had to be 
                <span style="font-weight:bold; color:#8da0cb">Ventilated</span>, and 
                <span style="font-weight:bold;">8</span> 
                <span style="font-weight:bold; color:#e78ac3">died</span>.
            </p>"""
        st.markdown(t, unsafe_allow_html=True)

    with cols[3]:
        #st.markdown("""Out of a __100__ Patients in this group, __18__ patients were admissioned to the __ICU__, __12__ had to be __ventilated__ and __25 died__.""")
        t = """<p>
                Out of a <span style="font-weight:bold;">100</span> Patients in this group, 
                <span style="font-weight:bold;">18</span> patients were admissioned to the 
                <span style="font-weight:bold; color:#fc8d62">ICU</span>, 
                <span style="font-weight:bold;">12</span> patients had to be 
                <span style="font-weight:bold; color:#8da0cb">Ventilated</span>, and 
                <span style="font-weight:bold;">25</span> 
                <span style="font-weight:bold; color:#e78ac3">died</span>.
            </p>"""
        st.markdown(t, unsafe_allow_html=True)

    with cols[4]:
        #st.markdown("""Out of a __100__ Patients in this group, __55__ patients were admissioned to the __ICU__, __46__ had to be __ventilated__ and __49 died__.""")
        t = """<p>
                Out of a <span style="font-weight:bold;">100</span> Patients in this group, 
                <span style="font-weight:bold;">55</span> patients were admissioned to the 
                <span style="font-weight:bold; color:#fc8d62">ICU</span>, 
                <span style="font-weight:bold;">46</span> patients had to be 
                <span style="font-weight:bold; color:#8da0cb">Ventilated</span>, and 
                <span style="font-weight:bold;">49</span> 
                <span style="font-weight:bold; color:#e78ac3">died</span>.
            </p>"""
        st.markdown(t, unsafe_allow_html=True)

def plot_preds(preds):
    #st.write("FOFOF")
    #st.write(preds.columns)
    #st.write(preds.values)
    values = preds.iloc[0].values
    maxind = values.argmax()
    pull=[0, 0, 0, 0, 0 , 0, 0]
    pull[maxind] = 0.2
    #st.write(values.reshape((1,7)))
    fig = go.Figure(
        data=[go.Pie(labels=preds.columns, values=values, hole=0.45, pull=pull)])
    st.plotly_chart(fig,use_container_width=True)

def plot_preds_pyechart(preds):
    labels = preds.columns
    values = preds.round(3).iloc[0].values
    values = values*100

    pie = (
        Pie()
        .add("Outcome", [list(z) for z in zip(labels,values)], radius=(40,75)
                #labels, values, radius=[40, 75],
                #label_text_color=None, 
                #is_label_show=True,
                #legend_orient='vertical', legend_pos='left'
        )
        .set_global_opts(
                title_opts=opts.TitleOpts(title="Predictions"),
                legend_opts=opts.LegendOpts(orient="vertical", pos_right=1)
        )
        .set_series_opts(label_opts=opts.LabelOpts(is_show=True, color=None),
                        linestyle_opts=opts.LineStyleOpts(is_show=True, color=None)
        )
    )
    return pie

@st.experimental_memo
def plot_preds_echart(preds):
    labels = preds.columns
    values = preds.round(3).iloc[0].values
    values = np.round(values*100,3)
    #labels = [f"{label} - {value}%" for label, value in zip(labels,values)]

    # ICU 1:"#fc8d62", VENT 2:"#8da0cb", DECEASED 3:"#e78ac3",
    cmap = {"Hospitalized (only)": "#edf8e9", "ICU": "#fc8d62", "Ventilated": "#8da0cb", "Deceased":"#e78ac3",
            "ICU + Ventilated": "#c4644f", "Deceased + Ventilated": "#b98ac1", "Deceased + ICU + Ventilated":"#941e67"}
    data = [{"value": val, "name": label, "itemStyle":{"color":cmap[label] if label in cmap.keys() else "green"}} for val, label in zip(values,labels)]
    #print(data)
    options = {
        "title": {
            "text": 'Predictions',
            #"left": 'center',
            "top": "5%",
            "textStyle": {
            "color": '#000',
            "fontWeight": '500',
            "fontSize": 20
            }
        },
        "tooltip": {"trigger": "item"},
        "legend": 
            {"bottom": "0%",
            "left": "0%",
            "orient": "vertical",
            "itemStyle": {
                    #"borderRadius": 6,
                    "borderColor": "#777",
                    "borderWidth": 1,
                },
            },
        "series": [
            {
                "name": "Outcome",
                "type": "pie",
                "radius": ["40%", "70%"],
                "avoidLabelOverlap": False,
                "itemStyle": {
                    "borderRadius": 6,
                    "borderColor": "#777",
                    "borderWidth": 1,
                },
                "label": {
                    "show": True, "color": None, "fontSize": "14",
                    "formatter": '{b} {c}%',
                    },
                "emphasis": {
                    "label": {"show": True, "fontSize": "16", "fontWeight": "bold"}
                },
                "labelLine": {"show": True},
                "data": data,
            }
        ],
    }
    return options

#@st.experimental_memo
def predictions(_ehd, _prettycols, cluster_means):
    inputs = st.session_state["user_inputs"]
    patient = create_patient_record(inputs, _ehd.columns, cluster_means)
    model = st.session_state["model"]
    preds = pd.DataFrame(model.predict_proba(patient), columns=_prettycols)
    return preds

@st.experimental_memo
def dummies(_prettycols):
    _dummies = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    preds = pd.DataFrame(_dummies, columns=_prettycols)
    return preds

@st.experimental_memo
def prepare_data(_ld):
    _st = time.time()

    ld = _ld
    _ehd = ld.process_features(return_df=True, normalize_dates=False)
    _ehd = _ehd.drop(ld._targets, axis=1, errors="ignore")
    
    # load clusters, make cluster 3->2 and 2->3
    _clusters = ld.patient_clusters
    _clusters[_clusters==3] = 5
    _clusters[_clusters==2] = 3
    _clusters[_clusters==5] = 2

    # create plot_df
    plot_df = _ehd.copy()
    plot_df["Group"] = _clusters
    plot_df["Outcome"] = ld.ml_target_df["Outcome"]

    # load targets and other info
    plot_df[ld._targets] = ld.target_df
    plot_df["Age"] = ld.raw_ehd["age.splits"]
    plot_df["Days Hospitalized"] = ld.raw_ehd["length_of_stay"]

    # count targets per cluster
    _ehd_sums = plot_df.groupby(by=["Group"]).sum()[ld._targets]

    props = (_ehd_sums.divide(plot_df.groupby("Group").count().iloc[:,0], axis=0)*100).round()

    icons = {"last.status": "tomb.png",
            "was_ventilated": "vent_patient.png",
            "is_icu": "icu_patient.png"}

    # sample props from dataframe to get some data
    resampled_df = pd.DataFrame()
    for colname, row in props.iteritems():
        for n, _cluster in zip(row, row.index):
            _df = plot_df[plot_df[colname] == 1].query("Group == @_cluster").sample(int(n), random_state=42).copy()
            _df["Hospitalization Outcome"] = colname # safe reason for sample
            _df["icon"] = icons[colname]
            resampled_df = resampled_df.append(_df, ignore_index=True)

    resampled_df["Hospitalization Outcome"].replace({"last.status": "Deceased",
                                                "was_ventilated": "Ventilated",
                                                "is_icu": "Intensive Care Unit"},
                                                 inplace=True)
    
    logging.info("Running {} took: {}".format("prepare_data", round(time.time()-_st, 3)))    
    #print("LEN CLSUTER 1: ", len(_clusters), np.unique(_clusters))
    return _ehd, _clusters, resampled_df
 

def build_overview(ld):
    topc1, topc2 = st.columns([7,2.0])
    with topc1:
        st.subheader("Clusters")
        textual = st.checkbox("Display Textual Description instead of infografic", value=False)
    with topc2:
        legend_placeholder = st.empty()
    
    _ehd, _clusters, resampled_df = prepare_data(ld)

    cluster_means = prepare_constants(_ehd, _clusters)
    #st.dataframe(cluster_means)
    
    
    if textual:
        display_cluster_help()
    else:
        image1 = Image.open('images/icons/group1.png')
        image2 = Image.open("images/icons/group2.png")
        image3 = Image.open("images/icons/group3.png")
        image4 = Image.open("images/icons/group4.png")
        #riskarrow = Image.open("../icons/legendarrow.png")
        legend = Image.open("images/icons/legend.png")
        with legend_placeholder:
            st.image(legend, caption="Legend")
        c0, c1, c2, c3, c4 = st.columns([0.01, 1,1,1,1])
        with c1:
            st.markdown("__Group 1 - Healthy Young to Middle Aged:__")
            st.image(image1)
        with c2:
            st.markdown("__Group 2 - Less Healthy Young to Middle aged:__")
            st.image(image2)
        with c3:
            st.markdown("__Group 3 - Elderly High Risk:__")
            st.image(image3)
        with c4:
            st.markdown("__Group 4 - High Risk:__")
            st.image(image4)

    st.markdown("---")

    _c1, _c2 = st.columns([0.02, 1])
    with _c2:
        #st.pyplot(simple_scatter(resampled_df))
        st.image("images/scatter_simple_small.png", use_column_width=True)
        #st.image(simple_scatter(resampled_df, figsize=(12,6)))
    display_cluster_per100()

    st.markdown("---")
    st.subheader("Outcome Prediction")
    #st.dataframe(_ehd)

    st.markdown("""Here you can input characteristis of yourself or somebody else, for instance a family member, or just play around with different inputs. 
From the given inputs a cluster is either _automatically_ selected or can be chosen _manually_. This data is used to fill in laboratory observations in order to create a prediction.
The underlying data consits of a total of ___1279___ patients all hospitalized during the first Covid-19 wave in 2020 in Stony Brooks, New York. 
    """)
    
    #build_prediction()

    _prettycols = ['Hospitalized (only)',
        'ICU', 'Ventilated', 'ICU + Ventilated',
        'Deceased', 'Deceased + Ventilated',
        'Deceased + ICU + Ventilated']
    with st.form("prediction-inputs"):
        ## row1
        col1, col2, col3, col4, col5, col6, col7 = st.columns([1.5,1.5,1.5,1.5,1.5,1.5,1.5])
        with col1:
            #age = st.number_input("Pick patients age", step=1, 
            #min_value=18, max_value=99, value=35)
            age = st.selectbox('Select Age', ('18 - 59', '59 - 74', '74 - 90'))
            diabetes = st.checkbox("Diabetes")
        
        with col2:
            gender = st.selectbox('Select gender', ('Male', 'Female'))
            htn = st.checkbox("High Blood Pressure")

        with col3:
            height = st.number_input("Patients height (cm)", step=1, 
                                    min_value=100, max_value=220, value=180)
            kidney_therapy = st.checkbox("Kidney replacement therapy")
        with col4: 
            weight = st.number_input("Patients weight (kg)", step=1, 
                                    min_value=35, max_value=250, value=70)
            kidney_transplant = st.checkbox("Kidney transplant")
        with col5:
            smoking_status = st.selectbox('Smoking Status', ('Never',
                'Former', 'Current', 'No Answer'))
            malignancies = st.checkbox("Malignancies")
        with col6:
            hfef_v = st.selectbox("Heart Failure", options=["No", "CAD", "HFpEF", "HFrEF"], help="""- __CAD__: _Coronary Artery Disease_
- __HFpEF Heart failure with preserved ejection fraction__: Heart failure is diagnosed but the pumping is still valid.  
- __HFrEF Heart failure with reduced ejection fraction__: Hearts pumping function is reduced.""")

            copd_v = st.checkbox("Chronic Obstructive Pulmonary Disease",
                    help="Chronic, progressive lung disease characterized by long-term respiratory symptoms and airflow limitation.")

        with col7: 
            #print("LEN CLUST: ", len(_clusters))
            cluster = st.selectbox('Select Group',
                ["Automatic"] + [i for i in np.unique(_clusters)])
            st.write(" ")
            st.write(" ")
            st.write(" ")
            submit = st.form_submit_button("Predict")
            if submit:
                inputs, verbose_inputs = _form_patient_inputs(age, gender, height, weight,
                             smoking_status, hfef_v, diabetes, htn, kidney_therapy,
                             kidney_transplant, malignancies, copd_v, cluster)
                print(inputs)
                
                st.session_state["user_inputs"] = inputs
                st.session_state["cluster_method"] = inputs["cluster"]
                st.session_state["user_inputs_verbose"] = verbose_inputs
                st.session_state["predicted-data"] = predictions(_ehd, _prettycols, cluster_means)

    c1, c_break, c2 = st.columns([1.5,0.1,2])

    
    with c1:
        placeholder_cluster = st.empty()
        st.subheader("Similar Patients")
        placeholder_neighbors = st.empty()
        placeholder_neighbors_info = st.empty()
    with c2:
        placeholder_plot=st.empty()
    
    
    data = st.session_state["predicted-data"] if "predicted-data" in st.session_state else dummies(_prettycols)
    #print(data)
    with placeholder_plot:
        st_echarts(plot_preds_echart(data), height="300px", key="PieResults")

    with placeholder_cluster:
        if "patient-infered-cluster" in st.session_state:
            if st.session_state["cluster_method"] == "Automatic":
                st.info("Automatic group assignment suggests __Group {}__ for your inputs.".format(st.session_state["patient-infered-cluster"]))
                #st.markdown("Automatic group assignment suggests __Group {}__ for your inputs.".format(st.session_state["patient-infered-cluster"]))
            else:
                st.info("Predictions supported by your Group selection: {}".format(st.session_state["cluster_method"]))
        else:
            st.markdown("")

    with placeholder_neighbors:
        if "user_inputs" in st.session_state:
            neighbors = _ehd.iloc[st.session_state["patient-infered-neighbors"]][st.session_state["input-columns"]]
            neighbors.columns = st.session_state["pretty-input-columns"]
            neighbors = neighbors.round(3) # this fixed errors for 1.0 being not replaced, cuz it was 1.000000002
            neighbors = neighbors.replace({1:"Yes", 0:"No", "1.0":"Yes", "1.0":"Yes",
                            1.0:"Yes", 0.0:"No",
                            0.1:"No", 0.2:"No", 0.3:"No", 0.4:"No", 
                            0.5:"Yes", 0.6:"Yes", 0.7:"Yes", 0.8:"Yes", 0.9:"Yes"})
            print(neighbors.Malignancies)
            #print(neighbors.COPD.values[0])
            #print(neighbors.COPD.dtype)
            #st.dataframe(neighbors.COPD)
            neighbors["Group"] = _clusters[st.session_state["patient-infered-neighbors"]]
            _columns = sorted(neighbors.columns)
            neighbors = neighbors.reindex(_columns, axis=1)
            neighbors["Outcome"] = ld.ml_target_df["Outcome"].iloc[st.session_state["patient-infered-neighbors"]]
            neighbors = neighbors[["Outcome"] + _columns]
            neighbors.set_index(ld.raw_ehd["to_patient_id"].iloc[st.session_state["patient-infered-neighbors"]], inplace=True)
            
            neighbors = neighbors.reset_index().rename({"to_patient_id":"Patient ID"},axis=1)
            print(neighbors)
            #print(neighbors.reset_index().rename({"to_patient_id":"Patient ID"},axis=1))

            #neighbors.set_index()
            st.dataframe(neighbors)
            #st.dataframe(neighbors[st.session_state["input-columns"]])
        else:
            st.markdown("You will find some similar patients here once you have put in some data above!")
    with placeholder_neighbors_info:
        if "user_inputs" in st.session_state:
            st.markdown("If interested, you can look up those patients with their _Patient ID_ in the __Medical Experts__ Page!")
            


