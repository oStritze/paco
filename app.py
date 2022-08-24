from matplotlib.ft2font import LOAD_NO_BITMAP
from src.preprocessing import process_features
from src.segmentation.models import PretrainedUNet, tuned_PretrainedUNet
import torch
import os
import pandas as pd
import streamlit as st
from src.streamlit_tabs import landingpage, general, medical, analytics
from src.data.lungdataset import LungData
import logging, inspect
import time

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


### SET X-RAYS PARENT FOLDER
rootdir = "/Volumes/Samsung_T5/MA/manifest-1641816674790/subsample_thresh1_A"

### DEBUG MODE ###
logging.basicConfig(level=logging.DEBUG)

st.set_page_config(
    page_title="PACO",
    #page_icon="💉",
    page_icon=":microbe:",
    layout='wide'
    )

hide_st_style = """
            <style>
                footer {visibility: hidden;}
            </style>
            """
# hide_st_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             header {background: rgba(255,255,255,0.);
#                   visibility: hidden;}
#             </style>
#             """
st.markdown(hide_st_style, unsafe_allow_html=True)
_padding_sides = 2.5
st.markdown(f"""
    <style>
        /* streamlit 1.3.0 */
        .reportview-container .main .block-container{{
            padding-top: 0rem;
            padding-left: {_padding_sides}rem;
            padding-right: {_padding_sides}rem;
        }}
        /* streamlit 1.4.0 */
        .block-container{{
            padding-top: 0rem;
            padding-left: {_padding_sides}rem;
            padding-right: {_padding_sides}rem;
        }}
        /* not working */
        stImage caption{{
            font-size: 20px
        }}
    </style>""",
    unsafe_allow_html=True,
)

st.markdown(
    '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">',
    unsafe_allow_html=True,
)

# Setting the background for the header to fully transparent. 
# The .css class hits finally - dunno if i have to change this, looks randomized. 
# Hierarchy: https://www.w3schools.com/css/css_specificity.asp /*
st.markdown(
    """
        <style>
            header {background: rgba(1,1,1,0.5);}
            #stHeader {background: rgba(1,1,1,0.5);}
            .css-k0sv6k {background: rgba(255,255,255, 0.0);} /* header */
            /* header {visibility: hidden;} */
            .css-nlntq9 a {color: rgba(252, 90, 80, 1); font-weight: 600;} /* nav-tabs */
            .css-aticvm {background-color: rgba(252, 90, 80, 0.75);} /* form submit button */
            .css-aticvm:hover {color: rgb(255,255,255)}
        </style>
    """, unsafe_allow_html=True
)

query_params = st.experimental_get_query_params()
tabs = ["Home", "Medical Experts", "Analytics", "General Population"]

###############################################################################
################################# Tab Handling ################################
###############################################################################
if "tab" in query_params:
    active_tab = query_params["tab"][0]
else:
    active_tab = "Home"

if active_tab not in tabs:
    st.experimental_set_query_params(tab="Home")
    active_tab = "Home"

li_items = "".join(
    f"""
    <li class="nav-item">
        <a class="nav-link{' active' if t==active_tab else ''}" href="/?tab={t}" target="_self" rel="">{t}</a>
    </li>
    """
    for t in tabs
)
tabs_html = f"""
    <ul class="nav nav-tabs">
    {li_items}
    </ul>
"""

### Custom Header
h1, h2 =  st.columns([3, 6])
with h1:
    st.title("COVID-19 Prediction", anchor="title")
with h2:
    st.markdown(tabs_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

###############################################################################
############################## Preprocessing ##################################
###############################################################################

#@st.cache
@st.experimental_memo
def load_data(rootdir:str="/Volumes/Samsung_T5/MA/manifest-1641816674790/subsample_thresh1_A"):
    _st = time.time()
    ld = LungData(rootdir, clusters_fpath="data/clusters.pkl")
    medical_df = ld.raw_ehd
    image_df = ld.as_dataframe()
    #ld.as_dataframe()
    logging.info("Ran {}, took {}s".format(inspect.currentframe().f_code.co_name,
            round(time.time()-_st, 3)))
    return ld, medical_df, image_df



#st.dataframe(resdf)

@st.experimental_memo
def _preprocess_data(raw_df):
    _df = process_features(raw_df, remove_identifiers=False)
    
    _df["last.status"].replace({"discharged":0, "deceased":1}, inplace=True)
    targets = ["last.status", "is_icu", "was_ventilated", "length_of_stay"]

    df_targets = _df[targets]
    df = _df.drop(targets + ["to_patient_id", "covid19_statuses", "visit_start_datetime", "invasive_vent_days"], axis=1)
    return df, df_targets, targets

@st.experimental_memo
def preprocess_data(_ld):
    _st = time.time()
    df = _ld.process_features(return_df=True)

    targets = ["last.status", "is_icu", "was_ventilated"]
    df_targets = df[targets]

    df.drop(targets + ["visit_start_datetime", "invasive_vent_days"], axis=1, inplace=True, errors="ignore")
    logging.info("Ran {}, took {}s".format(inspect.currentframe().f_code.co_name,
        round(time.time()-_st, 3)))
    return df, df_targets, targets

@st.experimental_memo
def load_segmentation_model(model_path:str="models/lung_seg/unet-6v.pt"):
    _st = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if model_path.startswith("t_"):
        unet = tuned_PretrainedUNet(
            in_channels=1,
            out_channels=2, 
            batch_norm=True, 
            #upscale_mode="bilinear"
        )
    else:
        unet = PretrainedUNet(
            in_channels=1,
            out_channels=2, 
            batch_norm=True, 
            upscale_mode="bilinear"
        )
    unet.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    unet.to(device)
    logging.info("Ran {}, took {}s".format(inspect.currentframe().f_code.co_name,
        round(time.time()-_st, 3)))
    return unet, device

@st.experimental_memo
def prepare_overview_df(images_df, medical_df, _ld):
    _st = time.time()
    _d = images_df.patient.value_counts()
    _df = pd.DataFrame(data=_d)
    _df.rename({"patient":"Nr of X-rays"}, axis=1, inplace=True)
    _mdf = medical_df.copy()
    _mdf["y"] = _ld.ml_target_df["y"]
    _mdf["Outcome"] = _ld.ml_target_df["Outcome"]
    for i, row in _mdf.iterrows():
        _df.at[row["to_patient_id"], ["Days Hospitalized"]] = row["length_of_stay"]
        _df.at[row["to_patient_id"], ["y"]] = row["y"]
        _df.at[row["to_patient_id"], ["Outcome"]] = row["Outcome"]
    _df["Outcome"].replace({"ICU + Vent": "ICU + Ventilated"}, inplace=True)
    _df = _df.reset_index().rename({"index": "Patient"}, axis=1) #reset to be able to access the patient-ids in plotly
    _df = _df.sort_values(by="y", ascending=True)
    logging.info("Ran {}, took {}s".format(inspect.currentframe().f_code.co_name,
        round(time.time()-_st, 3)))
    return _df

@st.experimental_memo
def load_radiomics(_ld):
    rdf = pd.read_csv("data/radiomics/thresh1_radiomics_all_stacked.csv")
    rdf = _ld.prepare_multiclass_for_radiomics(rdf, medical_df, verbose=False)
    rdfs_cleaned = pd.read_csv("data/radiomics/radiomics_cleaned_targets.csv")
    return rdf, rdfs_cleaned



###############################################################################
################################# Load Static #################################
###############################################################################
#patients, dcm, masks, external_available, raw_df, resdf = _load_data()
ld, medical_df, image_df = load_data(rootdir=rootdir)
model, device = load_segmentation_model()
df_pp, df_targets, targets = preprocess_data(ld)
prepared_df = prepare_overview_df(image_df, medical_df, ld)


#st.subheader("Testing App", anchor="Title")



if active_tab == "Home":
    _st = time.time()
    landingpage.build(medical_df)

    # st.write("Old method: ", raw_df.shape)
    # st.dataframe(raw_df)
    # st.slider(
    #     "Does this get preserved? You bet it doesn't!",
    #     min_value=0,
    #     max_value=100,
    #     value=50,
    # )
    logging.info("Building __{}__ took {}s".format(active_tab, round(time.time()-_st, 3)))


###############################################################################
################################# Medical #####################################
###############################################################################
elif active_tab == "Medical Experts":
    _st = time.time()
    #medical.build(raw_df, resdf, patients, dcm, external_available)
    #medical.overview(image_df, medical_df, ld)
    st.markdown("---")
    medical.build(image_df, medical_df, ld, prepared_df, model, device)
    logging.info("Building __{}__ took {}s".format(active_tab, round(time.time()-_st, 3)))

###############################################################################
################################# Analytics ###################################
###############################################################################

elif active_tab == "Analytics":
    rdf, rdfs_cleaned_targets = load_radiomics(ld)
    _st = time.time()
    #analytics.build_clustering(raw_df)
    st.markdown("---")
    analytics.overview_draft(ld, model, device, df_pp, rdf, rdfs_cleaned_targets)
    logging.info("Building __{}__ took {}s".format(active_tab, round(time.time()-_st, 3)))

###############################################################################
################################# General #####################################
###############################################################################

elif active_tab == "General Population":
    _st = time.time()
    general.build_overview(ld)

    logging.info("Building __{}__ took {}s".format(active_tab, round(time.time()-_st, 3)))

    # option = st.radio(
    #  'DEBUG Choose display method',
    #  ('Interactive', 'Non-Int'))
    # if option == "Non-Int":
    #     general.build(df, df_targets, targets)
    # else:
    #     general.build_interactive(df, df_targets, targets)
    
    # general.build_predictor(df, df_targets, targets)


elif active_tab == "About":
    _st = time.time()
    st.write("This page was created as a hacky demo of tabs")
    logging.info("Building __{}__ took {}s".format(active_tab, round(time.time()-_st, 3)))

else:
    st.error("Something has gone terribly wrong.")