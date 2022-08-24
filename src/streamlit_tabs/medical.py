from pydicom import dcmread
import streamlit as st
from src.utils import sel_random_patient, retrieve_image
from src.segmentation import segmentation, mask_cleaning
import os, sys, logging, time, inspect, math
import numpy as np
import pandas as pd
import torch, torchvision
from PIL import Image
from skimage.io import imread
from skimage.color import rgb2gray
import plotly.express as px

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
from streamlit_echarts import st_echarts, st_pyecharts

import radiomics
import SimpleITK as sitk
import joblib


#@st.experimental_memo
def set_constants():
    st.session_state["rad-settings"] = {'binWidth': 25,
                            'interpolator': sitk.sitkBSpline,
                            'resampledPixelSpacing': None}
    st.session_state["featureClasses"] = [radiomics.firstorder.RadiomicsFirstOrder, radiomics.shape2D.RadiomicsShape2D,
                    radiomics.glcm.RadiomicsGLCM, radiomics.glszm.RadiomicsGLSZM, radiomics.glrlm.RadiomicsGLRLM,
                    radiomics.ngtdm.RadiomicsNGTDM, radiomics.gldm.RadiomicsGLDM]
    st.session_state["rad-dropcols"] = ['MeanAbsoluteDeviation_l', 'Median_l', 'RobustMeanAbsoluteDeviation_l', 'RootMeanSquared_l', 'TotalEnergy_l', 'Uniformity_l', 'Variance_l', 'MaximumDiameter_l', 'PixelSurface_l', 'ClusterProminence_l', 'ClusterTendency_l', 'DifferenceEntropy_l', 'DifferenceVariance_l', 'Id_l', 'Idm_l', 'Imc2_l', 'InverseVariance_l', 'JointAverage_l', 'JointEnergy_l', 'MCC_l', 'MaximumProbability_l', 'SumAverage_l', 'SumEntropy_l', 'SumSquares_l', 'ZoneVariance_l', 'GrayLevelNonUniformityNormalized.1_l', 'GrayLevelVariance.1_l', 'HighGrayLevelRunEmphasis_l', 'RunPercentage_l', 'RunVariance_l', 'ShortRunEmphasis_l', 'DependenceEntropy_l', 'DependenceNonUniformity_l', 'GrayLevelVariance.2_l', 'HighGrayLevelEmphasis_l', 'LargeDependenceEmphasis_l', 'LowGrayLevelEmphasis_l', 'SmallDependenceEmphasis_l', 'MeanAbsoluteDeviation_r', 'Median_r', 'RobustMeanAbsoluteDeviation_r', 'RootMeanSquared_r', 'TotalEnergy_r', 'Variance_r', 'MaximumDiameter_r', 'PixelSurface_r', 'ClusterProminence_r', 'ClusterTendency_r', 'DifferenceAverage_r', 'DifferenceEntropy_r', 'DifferenceVariance_r', 'Id_r', 'Idm_r', 'InverseVariance_r', 'JointAverage_r', 'JointEnergy_r', 'MCC_r', 'SumAverage_r', 'SumEntropy_r', 'SumSquares_r', 'ZoneVariance_r', 'GrayLevelNonUniformityNormalized.1_r', 'GrayLevelVariance.1_r', 'HighGrayLevelRunEmphasis_r', 'RunPercentage_r', 'RunVariance_r', 'ShortRunEmphasis_r', 'DependenceEntropy_r', 'GrayLevelVariance.2_r', 'HighGrayLevelEmphasis_r', 'LargeDependenceEmphasis_r', 'LowGrayLevelEmphasis_r', 'SmallDependenceEmphasis_r']

    st.session_state["medical-rad-model"] = joblib.load("models/radiomics/svm_roc_auc_trained_on_all.pkl")


def predict_radiomics(new_obs):
    if "medical-rad-model" in st.session_state:
        model = st.session_state["medical-rad-model"]
    else:
        model = joblib.load("models/radiomics/svm_roc_auc_trained_on_all.pkl")
    #print(new_obs.columns.tolist())
    _prettycols = ['Hospitalized (only)',
    'ICU', 'Ventilated', 'ICU + Ventilated',
    'Deceased', 'Deceased + Ventilated',
    'Deceased + ICU + \nVentilated']
    #print(to_drop)
    try:
        new_obs = new_obs.drop(st.session_state["rad-dropcols"], axis=1)
        #print(new_obs)
        #print(new_obs.max())
        preds = pd.DataFrame(model.predict_proba(new_obs), columns=_prettycols)
        logging.info("NEW PREDICTION: {}".format(preds))
        return preds
    except KeyError as e:
        newcols = new_obs.columns.tolist()
        dropcols = st.session_state["rad-dropcols"]
        diff = set(newcols) - set(dropcols)
        negdiff = set(dropcols) - set(newcols)
        #print("DIFF: ", diff)
        #print("NEGDIFF: ", negdiff)
        logging.info("ERROR PREDICTING -- columns missmatch ", preds)
        #st.dataframe(new_obs)
        return 0
    

def extract_radiomics(xray, mask_r, mask_l):
    _st = time.time()
    #print(xray.shape, xray.max())
    #xray = np.round(xray*255)
    #mask_r = np.round(mask_r*255)
    #mask_l = np.round(mask_l*255)
    if "rad-settings" not in st.session_state:
        featureClasses = [radiomics.firstorder.RadiomicsFirstOrder, radiomics.shape2D.RadiomicsShape2D,
                    radiomics.glcm.RadiomicsGLCM, radiomics.glszm.RadiomicsGLSZM, radiomics.glrlm.RadiomicsGLRLM,
                    radiomics.ngtdm.RadiomicsNGTDM, radiomics.gldm.RadiomicsGLDM]
        settings = {'binWidth': 25, 'interpolator': sitk.sitkBSpline, 'resampledPixelSpacing': None}
    else:
        settings = st.session_state["rad-settings"]
        featureClasses = st.session_state["featureClasses"]

    xray = sitk.GetImageFromArray(xray) 
    roi_left = sitk.GetImageFromArray(mask_l)
    roi_right = sitk.GetImageFromArray(mask_r)

    left_df = pd.DataFrame()
    right_df = pd.DataFrame()
    for fc in featureClasses:
        #print(fc, type(fc))
        feats_left = fc(xray, roi_left, **settings)
        feats_right = fc(xray, roi_right, **settings)

        res_left = feats_left.execute()
        res_right = feats_right.execute()

        _res_left = pd.DataFrame.from_dict(res_left, orient="index").T
        _res_right = pd.DataFrame.from_dict(res_right, orient="index").T

        ## MANUALLY RENAME THOSE COLUMSN BECAUSE I DONT KNOW WHAT THE FUCK PANDAS
        ### GOT EVEN PRETTIER WITH isinstance NOT WORKING!
        if fc.__name__ ==  "RadiomicsNGTDM":
            logging.debug("RENAMING NGTDM")
            _res_left.rename({"Contrast": "Contrast.1"}, axis=1, inplace=True)
            _res_right.rename({"Contrast": "Contrast.1"}, axis=1, inplace=True)
        elif fc.__name__ == "RadiomicsGLRLM":
            logging.debug("RENAMING GLRLM")
            _res_left.rename({"GrayLevelNonUniformity": "GrayLevelNonUniformity.1",
                    "GrayLevelNonUniformityNormalized": "GrayLevelNonUniformityNormalized.1",
                    "GrayLevelVariance": "GrayLevelVariance.1"}, axis=1, inplace=True)
            _res_right.rename({"GrayLevelNonUniformity": "GrayLevelNonUniformity.1",
                    "GrayLevelNonUniformityNormalized": "GrayLevelNonUniformityNormalized.1",
                    "GrayLevelVariance": "GrayLevelVariance.1"}, axis=1, inplace=True)
        elif fc.__name__ == "RadiomicsGLDM":
            logging.debug("RENAMING GLDM")
            _res_left.rename({
                    "GrayLevelVariance": "GrayLevelVariance.2"}, axis=1, inplace=True)
            _res_right.rename({
                    "GrayLevelVariance": "GrayLevelVariance.2"}, axis=1, inplace=True)


        left_df = pd.concat([left_df, _res_left], axis=1)
        right_df = pd.concat([right_df, _res_right], axis=1)


    left_df.columns = [i+"_l" for i in left_df.columns]
    right_df.columns = [i+"_r" for i in right_df.columns]
    merged_results = pd.concat([left_df, right_df], axis=1)
    logging.info("Function {}, took {}s".format(inspect.currentframe().f_code.co_name,
            round(time.time()-_st, 3)))
    return merged_results

def _slider_format(input:str):
    return f"{input} day(s) from most critical obs"

def _select_random_image(images_df):
    sample = images_df.sample()
    print("Current selection: ",sample["patient"].values, sample["png"].values, sample["mask_cleaned"].values)
    return sample["patient"].values[0], sample["png"].values[0], sample["mask_cleaned"].values[0], sample["delta_date"].values[0]

def _load_image(input:str):
    img = imread(input)
    img = Image.fromarray(img).convert("P")
    img = torchvision.transforms.functional.resize(img, (512, 512))
    return img

def _blend(img:str, mask:str):
    img = _load_image(img)
    mask = imread(mask, as_gray=True)
    mask_blend = (np.stack(( np.zeros((512,512)), mask, np.zeros((512,512)) ), axis=2) * 255).astype(np.uint8).reshape(512,512,3)
    #blended = Image.blend(img.convert("RGB"), Image.fromarray(mask), 0.2)
    blent = Image.blend(img.convert("RGB"), Image.fromarray(mask_blend), 0.2)
    return blent

def _blend_PIL(img:Image, mask):
    #print(type(img), type(mask))
    blent = np.zeros((512,512))
    if isinstance(mask, Image.Image):
        blent = Image.blend(img.convert("RGB"), Image.fromarray(mask), 0.2)
    elif isinstance(mask, np.ndarray):
        if mask.shape != (512,512):
            mask = mask[:, :, 0] # if still RGB, only take one channel
        mask_blend = (np.stack(( np.zeros((512,512)), mask, np.zeros((512,512)) ), axis=2) * 255).astype(np.uint8).reshape(512,512,3)
        blent = Image.blend(img.convert("RGB"), Image.fromarray(mask_blend), 0.2)
    return blent

def _process_upload(upload, preproc, model, device):
    _st = time.time()

    if preproc != "None":
        if preproc == "Blur":
            _preproc = segmentation.blur_adapthisteq
        elif preproc == "Median":
            _preproc = segmentation.median_adapthisteq
    elif preproc == "None":
        _preproc = None

    if upload.type in ["image/png", "image/jpg"]:
        logging.debug("READING PNG/JPG")
        img = Image.open(upload)
        img = np.array(img)
        #print(img.shape, img.max())
        if preproc is not "None":
            img = _preproc(img)
        img = np.round(img/img.max() * 255)
        img = Image.fromarray(img).convert("P")
    elif upload.type == "application/dicom":
        logging.debug("READING DICOM")
        img = dcmread(upload)
        img = img.pixel_array # max values are around 4000, scale to [0,1] first

        if preproc is not "None":
            img = _preproc(img)
        img = np.round(img/img.max() * 255)
        img = Image.fromarray(img).convert("P")
    
    img = torchvision.transforms.functional.resize(img, (512, 512))
    
    with torch.no_grad():
        origin = torchvision.transforms.functional.to_tensor(img) - 0.5
        origin = torch.stack([origin])
        origin = origin.to(device)
        out = model(origin)
        softmax = torch.nn.functional.log_softmax(out, dim=1)
        out = torch.argmax(softmax, dim=1)
        
        origin = origin[0].to("cpu")
        mask = out[0].to("cpu")
        maskn = (np.stack(   (np.zeros((512,512)), 
                            mask.numpy(),
                            np.zeros((512,512)) ), axis=2) *255).astype(np.uint8).reshape(512,512,3)
    #print(mask, mask.max())
    # clean mask from artifacts -> 2 contours only
    mask_cleaned, mask_r, mask_l = mask_cleaning.mask_separator(maskn, save_images=False, debug=False)

    blent = _blend_PIL(img, mask_cleaned) # blend
    logging.info("Image loading and segmentation took {}s".format(round(time.time()-_st, 3)))
    _st2 = time.time()
    radiomics_features = extract_radiomics(img, mask_r, mask_l)
    st.session_state["radiomics-record"] = radiomics_features
    #print(radiomics_features.shape)
    logging.info("Radiomics feat. extraction took {}s".format(round(time.time()-_st2, 3)))
    _st2 = time.time()
    preds = predict_radiomics(radiomics_features)
    logging.info("Predicting from picture input took {}s".format(round(time.time()-_st2, 3)))

    st.session_state["radiomics-record-prediction"] = preds
    logging.info("Function {}, took {}s".format(inspect.currentframe().f_code.co_name,
            round(time.time()-_st, 3)))

    return img, blent, preds

def plot_preds_bar_echart(preds):
    labels = preds.columns.values.tolist()
    values = preds.round(3).iloc[0].values
    values = np.round(values*100,3)
    print(labels, values)
    data = [{"value": val, "name": label} for val, label in zip(values,labels)]
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
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {
            "type": "shadow"
            }
        },
        "xAxis": {
            "type": "category",
            "data": labels,
            "axisLabel": {
                "interval": 0, #show all 
                "rotate": 0,
                #"align": "right",
            }
        },
        "yAxis":{
            "type": "value",
            "min": "0",
            "max": "100",
            "name": "Probability in %",
            "nameLocation":"center",
            "nameRotate":90,
            "nameGap":30,
        },
        "tooltip": {"trigger": "item"},
        "legend": 
            {"bottom": "0%",
            "left": "50%",
            "orient": "horizontal" 
            },
        "series": [
            {
                "name": "Outcome",
                "type": "bar",
                "data": data,
                "label": {
                    "show": False, "color": None, "fontSize": "14",
                    "formatter": '{b} {c}%',
                    },
                # "emphasis": {
                #     "label": {"show": True, "fontSize": "16", "fontWeight": "bold"}
                # },
                # "labelLine": {"show": True},
                "showBackground": "true",
                "backgroundStyle": {
                    "color": 'rgba(180, 180, 180, 0.2)'
                },
                "barWidth": "60%"
            }
        ],
    }
    return options


@st.experimental_memo
def dummies():
    _prettycols = ['Hospitalized (only)',
        'ICU', 'Ventilated', 'ICU + Ventilated',
        'Deceased', 'Deceased + Ventilated',
        'Deceased + ICU + \nVentilated']
    _dummies = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    preds = pd.DataFrame(_dummies, columns=_prettycols)
    return preds


def display_blend(image, mask):
    col1, col2, col3 = st.columns(3)
    #col1, col2, col3, col4, col5 = st.columns([1.5, 2, 2.5, 2, 1.5])
    with col1:
        st.image(_load_image(image),
            use_column_width="auto", caption="X-Ray")
    with col2:
        st.image(_blend(image, mask),
                use_column_width="auto", caption="Segmented Lungs")
    with col3:
        st.image(mask,
                use_column_width="auto", caption="Raw mask")
    

def display_chronic(patient, day, images_df, blend=True, range:int=5):
    _st = time.time()
    # create array for iterating images
    _idx = np.arange(0, range)-math.floor(range/2) 
    # _idx = [-2, -1, 0, 1, 2] 
    col1, col2, col3, col4, col5 = st.columns(
        [2.3, 2.4, 2.5, 2.4, 2.3]
        )
    cols = [col1, col2, col3, col4, col5]

    # safe patient images in df
    patient_images = images_df[images_df["patient"] == patient].sort_values(
                        by="delta_date", ascending=True).reset_index(drop=True)

    #st.dataframe(patient_images)
    # save current selected index by day for the patient
    selected_id = patient_images[patient_images["delta_date"] == day].index[0]

    # zip over cols and idx's, go before and beyond sel image by _idx array
    for col, idx in zip(cols, _idx):
        this_id = selected_id + idx
        day_caption = idx

        #st.write(patient_images.loc[[this_id]]["delta_date"])
        if 0 <= this_id <= patient_images.index.max(): # solving caption only
            day_caption = patient_images.loc[[this_id]]["delta_date"].values[0]
        if this_id < 0:
            #this_id = 0 #first possible index is always 0
            # first possible day is not always 0, can be negative
            #day_caption = patient_images.loc[[this_id]]["delta_date"].values[0]

            # continue skips the show-image part, which is pretty cool actually
            continue
        elif this_id > patient_images.index.max():
            #this_id = patient_images.index.max()
            #day_caption = patient_images["delta_date"].max()
            continue
            
        if blend:
            image = patient_images.loc[[this_id]]["png"].values[0]
            mask = patient_images.loc[[this_id]]["mask_cleaned"].values[0]
            col.image(_blend(image, mask),
                use_column_width="auto", caption="{} - Day: {}".format(patient, day_caption))
        else:
            image = _load_image(patient_images.loc[[this_id]]["png"].values[0]) # resize images
            col.image(image,
                use_column_width="auto", caption="{} - Day: {}".format(patient, day_caption))
    logging.info("Function {}, took {}s".format(inspect.currentframe().f_code.co_name,
        round(time.time()-_st, 3)))



#@st.experimental_memo
def _comp_overview_df(_df, checkbox_inputs, table_selection=None):
    #print("PDF shape", _df.shape)
    
    # for key, val in input -> Deceased:True
    for target_type, input in checkbox_inputs.items():
        if input == False: # if not include this type
            _df = _df[~_df["Outcome"].str.contains(target_type)]
    #print("df after checkboxes check: ", _df.shape)

    if table_selection is not None:
        print("table sel", table_selection.shape)
        _df = _df[_df["Patient"].isin(table_selection)]
    #print("comp_df shape: ", _df.shape)
    return _df


def _custom_pandas_style(row, current_selection):
    color = 'white'
    if row.index == current_selection:
        color = 'yellow'
    return ['background-color: %s' % color]*len(row.values)


def upload(model, device, disable_preproc_options=True, return_preds=False):
    
    col1, col2, col3, col4, col5 = st.columns([3, 0.5, 2, 2, 1])

    with col1:
        uploaded_file = st.file_uploader("", type=["png", "jpg", "dcm"])
        st.markdown("If you dont have one but want to try things out, you can download one [here](https://github.com/oStritze/thesis-sample-xrays).")
        if not disable_preproc_options:
            upload_preproc = st.radio(
                "Choose an optional preprocessing strategy",
                ('None', 'Blur', 'Median'), index=2)
        else:
            upload_preproc = "Blur"
    with col3:
        if uploaded_file is not None:
            uploaded_img, blent, preds = _process_upload(uploaded_file, upload_preproc, model, device)
            st.image(uploaded_img, use_column_width="always")
        else:
            st.image("images/placeholder-image.webp", width=250, caption="Placeholder", use_column_width="always")
    with col4:
        if uploaded_file is not None:
            st.image(blent, use_column_width="always")
        else:
            st.image("images/placeholder-image.webp", width=250, caption="Placeholder", use_column_width="always")
    # if uploaded_file is not None:
    #     img = Image.open(uploaded_file)
    #     img = np.array(img)
    #     #f = rgb2gray(f)
    #     st.image(img, width=200)
    #     img = Image.fromarray(img).convert("P")
    #     img = torchvision.transforms.functional.resize(img, (512, 512))
    #     st.image(img, width=200)
    if return_preds and uploaded_file is not None:
        return preds
    elif return_preds and uploaded_file is None:
        return None

def build(images_df, medical_df, ld, prepared_df, model, device):
    #_df = images_df.copy()
    #overview(images_df, medical_df, ld)

    set_constants()

    _st = time.time()

    row0_col01, row0_col0, row0_col1, row0_col2, row0_col3, row0_col4 = st.columns([5, 0.5, 1, 1, 1, 1])
    with row0_col01:
        st.subheader("Overview", anchor="Overview")
    with row0_col0:
        st.write(" ")
        st.write("Include:")
    with row0_col1:
        st.write(" ")
        _checkbox_include_deceased = st.checkbox("Deceased", value=True)
    with row0_col2:
        st.write(" ")
        _checkbox_include_icu = st.checkbox("ICU", value=True)
    with row0_col3:
        st.write(" ")
        _checkbox_include_vent = st.checkbox("Ventilated", value=True)
    with row0_col4:
        st.write(" ")
        _checkbox_include_hosp = st.checkbox("Hospitalized", value=True)

    checkbox_inputs_row0 = {"Deceased": _checkbox_include_deceased, "ICU": _checkbox_include_icu,
                            "Vent":_checkbox_include_vent, "Hospitalized": _checkbox_include_hosp}
    
    if "table_selection" in st.session_state:
        #print("TABLE SELECTION PRESENT")
        ov_df = _comp_overview_df(prepared_df, checkbox_inputs_row0, st.session_state["table_selection"])
    else:
        ov_df = _comp_overview_df(prepared_df, checkbox_inputs_row0)

    logging.info("MEDICAL: ov_df took {}s".format(time.time()-_st))
    
    row1_col1, row1_col2 = st.columns([2 ,2])
    with row1_col1:
        #st.write(ov_df.shape)
        plotly_scatterplot_placeholder = st.empty()

    logging.info("MEDICAL: building scatter took {}s".format(time.time()-_st))
    # only show options for previous-selected patients
    _sub_df = medical_df[medical_df["to_patient_id"].isin(ov_df["Patient"])].sort_values(by="to_patient_id").reset_index(drop=True).copy()
    _sub_df_patients = _sub_df["to_patient_id"]

    #_df = _sub_df.set_index("to_patient_id")

    with row1_col2:

        patients_table_placeholder = st.empty()

        #st.dataframe(_sub_df["to_patient_id"].unique())
        #st.write(0 if "random_patient_idx" not in st.session_state \
        #                                else st.session_state["random_patient_idx"])
        
    row2_col1, row2_col2, row2_col3 = st.columns([1,1,2])

    with row2_col1:
        patient_selection_placeholder = st.empty()

    with row2_col2:
        st.write(" ")
        st.write(" ")

        if st.button("Select a Random Patient"):
            selected_patient, selected_image, selected_mask, delta = \
                _select_random_image(images_df[images_df["patient"].isin(_sub_df_patients)])
            
            _index = np.where(images_df[images_df["patient"].isin(_sub_df_patients)].patient.unique() == selected_patient)[0][0]
            #print("RANDOM: ", selected_patient, _index)
            st.session_state["selected-patient"] = selected_patient
            if 'random_patient_idx' not in st.session_state:
                st.session_state["random_patient_idx"] = int(_index)
            else:
                st.session_state["random_patient_idx"] = int(_index)
    
    with patient_selection_placeholder:
        selected_patient = st.selectbox('Select Patient Manually ',
                                    _sub_df["to_patient_id"].reset_index(drop=True),
                                    #index="A000936"\
                                    #    if "random_patient_idx" not in st.session_state \
                                    #    else st.session_state["random_patient_idx"],
                                    key="selected-patient"
            )

    logging.info("MEDICAL: rendered patient selection after {}s".format(time.time()-_st))
    with patients_table_placeholder:
        if "table_selection" in st.session_state:
            #print("TABLE SELECTION PRESENT")
            ov_df = _comp_overview_df(prepared_df, checkbox_inputs_row0, st.session_state["table_selection"])
        else:
            ov_df = _comp_overview_df(prepared_df, checkbox_inputs_row0)
        _sub_df = medical_df[medical_df["to_patient_id"].isin(ov_df["Patient"])].sort_values(by="to_patient_id").reset_index(drop=True).copy()
        _sub_df_patients = _sub_df["to_patient_id"]
        #print("--------table sub_df shape:", _sub_df.shape)
        if "selected-patient" in st.session_state:
            crs = st.session_state["selected-patient"]
        else:
            #print("ITS NOT IN IT")
            crs = 0
        # TODO: highlight selected patient
        ## this literally takes forever: 45s 
        # st.dataframe(_sub_df.set_index("to_patient_id").style.apply(
        #         lambda x: ['background: lightblue' if x.name == crs else '' for i in x], 
        #            axis=0), height=200)
        ## this is not working, method is not defined?
        # st.dataframe(_sub_df.set_index("to_patient_id").style.apply_index(
        #         lambda x: ['background: lightblue' if x == crs else '' for i in x]
        #         ), height=200)
        
        ## Previous Raw Column
        #st.dataframe(_sub_df.set_index("to_patient_id"), height=200)
        gb = GridOptionsBuilder.from_dataframe(_sub_df)
        crs_idx = _sub_df[_sub_df["to_patient_id"] == crs].index.values[0]
        #print("CRS: ", crs, crs_idx, type(crs_idx))
        
        #print(crs, crs_id, crs_id[0])
        gb.configure_selection('single', pre_selected_rows=[int(crs_idx)])
        response = AgGrid(
            _sub_df,
            editable=True,
            gridOptions=gb.build(),
            update_mode=GridUpdateMode(1),
            height=350, 
            data_return_mode="filtered",
            fit_columns_on_grid_load=False,
            theme="light"
        )
        #print(response["data"]["to_patient_id"].values)
        st.session_state["table_selection"] = response["data"]["to_patient_id"].values
        #ov_df = _comp_overview_df(prepared_df, checkbox_inputs_row0, st.session_state["table_selection"])


    logging.info("MEDICAL: rendered table after {}s".format(time.time()-_st))

    if "table_selection" in st.session_state:
        print("TABLE SELECTION PRESENT")
        ov_df = _comp_overview_df(prepared_df, checkbox_inputs_row0, st.session_state["table_selection"])
    else:
        ov_df = _comp_overview_df(prepared_df, checkbox_inputs_row0)

    with plotly_scatterplot_placeholder:
        cmap = {"Hospitalized (only)": "#edf8e9", "ICU": "#fc8d62", "Ventilated": "#8da0cb", "Deceased":"#e78ac3",
            "ICU + Ventilated": "#c4644f", "Deceased + Ventilated": "#b98ac1", "Deceased + ICU":"#c35065",
            "Deceased + ICU + Ventilated":"#941e67"}
        fig = px.scatter(ov_df, x="Days Hospitalized", y ="Nr of X-rays",
                color="Outcome", #color_discrete_sequence=list(cmap.values()), # TODO: Use better colormap here
                color_discrete_map=cmap,
                hover_name="Patient",
                #marginal_x="box",
                #marginal_y="box",
                )

        fig.for_each_trace(lambda t: t.update({"marker":{"size":10, "opacity":0.8, #"symbol":"hexagon",
                                            "line":{"width":2, "color":"DarkSlateGrey"}
                                            }}
                                        ))
        fig.update_layout(legend=dict(
                orientation="h",
                #yanchor="top",
                #y=-.17,
                y=1.2,
                xanchor="right",
                x=0.9,#1.0,
            ),
            height=340,
            margin=dict(
                l=0,
                r=10,
                b=30,
                t=0,
                pad=4
            ),
        )
        fig.update_xaxes(range=[-2, 100])
        fig.update_yaxes(range=[-2, 60], autorange=False)
        # fig.update_layout(
        #     modebar_remove=['zoom', 'pan', 'select', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale', 'displaylogo', 'lasso2d']
        # )
        st.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': False
        })


    st.markdown("---")

    rond_container = st.container()

    col1, col2, col3, col4, col5 = st.columns([1.5, 0.25, 5, 0.25, 1.25])

    with col1:
        st.markdown("Current Selection __{}__ is __{}__ and between __{}__ years old. The patient was hospitalized for __{}__ days. The patients last status was __{}__.".format(selected_patient,
                medical_df[medical_df.to_patient_id == selected_patient]["gender_concept_name"].iloc[0],
                medical_df[medical_df.to_patient_id == selected_patient]["age.splits"].iloc[0],
                medical_df[medical_df.to_patient_id == selected_patient]["length_of_stay"].iloc[0],
                medical_df[medical_df.to_patient_id == selected_patient]["last.status"].iloc[0],
                ), unsafe_allow_html=True)


    with col3: 
        _disable_slider = False
        opts = np.array(sorted(
            images_df[images_df["patient"] == selected_patient]["delta_date"].values))
        
        # for pats with <3 images, take the third entry for the middle image in the carousel
        if len(opts) > 3: 
            _startopt = np.argmin(abs(opts))+2
        elif 3 >= len(opts) > 1: # for pats with 2-3 images, take the 2nd entry
            # the argmin of abs here takes the 2nd because the first entry is usually -1
            # and the 2nd is then zero, which is min
            _startopt = np.argmin(abs(opts))
        elif len(opts) == 1:
            #print(opts)
            _disable_slider = True
            opts = np.append(opts, [1])
            _startopt = 0
        delta = st.select_slider(
                label="Select picture at observation point",
                options=opts,
                value=opts[_startopt], #TODO: gets killed with pats like this A352169 -- stil gets killed with A843980
                format_func=_slider_format,
                help="""This value defines the date delta in days between the most critical
                        state of the patient and the date of the X-Ray recording. The default
                        selection is the image that is the nearest to the most critical state.
                        """,
                disabled=_disable_slider
                )


    with col5:
        st.write("Blend segmentation")
        cb_blend = st.checkbox('', value=True,
         help="Un-check if the segmentation should not be shown")

    #display_blend(selected_image, selected_mask)

    with rond_container:
        display_chronic(selected_patient, delta, images_df, cb_blend)


    st.markdown("---")
    st.subheader("Upload own X-ray")
    upload(model, device)

    st.markdown("__Prediction__:")
    c1,c2 = st.columns([0.5,2])
    with c1:
        if "radiomics-record-prediction" in st.session_state:
            #st.write(st.session_state["radiomics-record-prediction"])
            st.write("")
        else: 
            st.write("Upload a picture!")
    with c2:
        preds_chart_placeholder = st.empty()

    data = st.session_state["radiomics-record-prediction"] if "radiomics-record-prediction" in st.session_state else dummies()
    #if "radiomics-record-prediction" in st.session_state:
    with preds_chart_placeholder:
        st_echarts(plot_preds_bar_echart(data), height="300px", key="BarResults")
        #st.write(st.session_state["radiomics-record-prediction"])
    
    st.markdown("---")
