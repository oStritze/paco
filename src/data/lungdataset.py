from aifc import Error
from ast import Assert
from multiprocessing.dummy import Pipe
from typing import Dict, List
import joblib
import pydicom
from skimage.io import imread 
import pandas as pd
import numpy as np
import os
import SimpleITK as sitk
import radiomics as rad
from PIL import Image
from sklearn.neighbors import kneighbors_graph
import torchvision
import datetime

import re

from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.cluster import DBSCAN, KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer


_default_path = "data/deidentified_overlap_tcia.csv.cleaned.csv_20210806.csv"

class LungData():
    _types = ["patient", "dcm", "png", "mask_cleaned", "mask_l", "mask_r", "date"]
    _settings = {'binWidth': 25,
            'interpolator': sitk.sitkBSpline,
            'resampledPixelSpacing': None}

    ml_ordering = ['Hospitalized (only)',
        'ICU', 'Ventilated', 'ICU + Vent',
        'Deceased', 'Deceased + ICU', 'Deceased + Ventilated',
        'Deceased + ICU + Ventilated']

    ml_ordering_short = ['Hospitalized',
        'ICU', 'Vent.', 'ICU + Vent.',
        'Deceased', 'Deceased + ICU', 'Deceased + Vent.',
        'Deceased + ICU + Vent.']

    ### my best developer work is to be found in the next 190 lines
    # all of those occur as numerical columns in the dataset as well
    _drop_categorical = ['blood_pH.above7.45', "blood_pH.between7.35and7.45", "blood_pH.below7.35", 
        'Chloride.above107', 'Chloride.between96and107', 'Chloride.below96',
        'Sodium.above145', 'Sodium.between135and145', 'Sodium.below135',
        'Potassium.above5.2', 'Potassium.between3.5and5.2', 'Potassium.below3.5',
        'Bicarbonate.above31', 'Bicarbonate.between21and31', 'Bicarbonate.below21',
        'A1C.over6.5', 'A1C.under6.5', 'A1C.6.6to7.9', 'A1C.8to9.9', 'A1C.over10',
        'Blood_Urea_Nitrogen.above20', 'Blood_Urea_Nitrogen.between5and20', 'Blood_Urea_Nitrogen.below5',
        'Creatinine.above1.2', 'Creatinine.between0.5and1.2', 'Creatinine.below0.5',
        'Troponin.above0.01',
        'D_dimer.above3000', 'D_dimer.between500and3000', 'D_dimer.below500',
        'procalcitonin.below0.25', 'procalcitonin.between0.25and0.5', 'procalcitonin.above0.5',
        'ferritin.above1k',
        'ESR.above30', # Erythrocyte sedimentation rate
        'Lymphocytes.under1k',
        'BMI.over30', 'BMI.over35',
        'SBP.above139', 'SBP.below120','SBP.between120and139', # systolic blood pressure
        'MAP.below65', 'MAP.between65and90', 'MAP.above90', # Oxygen saturation in Arterial blood by Pulse oximetry
        'eGFR.above60', 'eGFR.below30', 'eGFR.between30and60', # estimated Glomerular filtration rate -> estimate how well your kidneys are filtering certain agents
        'Aspartate.over40', 
        'HeartRate.over100', 
        'Alanine.over60',
        'temperature.over38',
        'pulseOx.under90', 'Respiration.over24',
        'visit_concept_name',
        'therapeutic.exnox.Boolean',
        'therapeutic.heparin.Boolean',
        'Other.anticoagulation.therapy',
        'days_prior_sx',
    ]

    _lab_features = ['Urine.protein', 'Microscopic_hematuria.above2', 'Proteinuria.above80',
       'Oxygen saturation in Arterial blood by Pulse oximetry',
       'Respiratory rate', 'Heart rate.beat-to-beat by EKG',
       'Systolic blood pressure',
       'Mean blood pressure by Noninvasive',
       'Leukocytes [#/volume] corrected for nucleated erythrocytes in Blood by Automated count',
       'Neutrophils [#/volume] in Blood by Automated count',
       'Lymphocytes [#/volume] in Blood by Automated count',
       'Sodium [Moles/volume] in Serum or Plasma',
       'Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma',
       'Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma by No addition of P-5\'-P',
       'Creatine kinase [Enzymatic activity/volume] in Serum or Plasma',
       'Lactate [Moles/volume] in Serum or Plasma',
       'Troponin T.cardiac [Mass/volume] in Serum or Plasma',
       'Natriuretic peptide.B prohormone N-Terminal [Mass/volume] in Serum or Plasma',
       'Procalcitonin [Mass/volume] in Serum or Plasma by Immunoassay',
       'Fibrin D-dimer DDU [Mass/volume] in Platelet poor plasma by Immunoassay',
       'Ferritin [Mass/volume] in Serum or Plasma',
       'C reactive protein [Mass/volume] in Serum or Plasma',
       'Hemoglobin A1c/Hemoglobin.total in Blood',
       'Potassium [Moles/volume] in Serum or Plasma',
       'Chloride [Moles/volume] in Serum or Plasma',
       'Bicarbonate [Moles/volume] in Serum or Plasma',
       'Urea nitrogen [Mass/volume] in Serum or Plasma',
       'Creatinine [Mass/volume] in Serum or Plasma',
       'Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (CKD-EPI)',
       'pH of Arterial blood adjusted to patient\'s actual temperature',
       'Erythrocyte sedimentation rate',
       'Glucose [Mass/volume] in Serum or Plasma',
       'Cholesterol in LDL [Mass/volume] in Serum or Plasma by calculation',
       'Cholesterol in VLDL [Mass/volume] in Serum or Plasma by calculation',
       'Triglyceride [Mass/volume] in Serum or Plasma',
       'Cholesterol in HDL [Mass/volume] in Serum or Plasma']

    _lab_features_with_codes = ['Urine.protein', 'Microscopic_hematuria.above2', 'Proteinuria.above80',
       '59408-5_Oxygen saturation in Arterial blood by Pulse oximetry',
       '9279-1_Respiratory rate', '76282-3_Heart rate.beat-to-beat by EKG',
       '8480-6_Systolic blood pressure',
       '76536-2_Mean blood pressure by Noninvasive',
       '33256-9_Leukocytes [#/volume] corrected for nucleated erythrocytes in Blood by Automated count',
       '751-8_Neutrophils [#/volume] in Blood by Automated count',
       '731-0_Lymphocytes [#/volume] in Blood by Automated count',
       '2951-2_Sodium [Moles/volume] in Serum or Plasma',
       '1920-8_Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma',
       '1744-2_Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma by No addition of P-5\'-P',
       '2157-6_Creatine kinase [Enzymatic activity/volume] in Serum or Plasma',
       '2524-7_Lactate [Moles/volume] in Serum or Plasma',
       '6598-7_Troponin T.cardiac [Mass/volume] in Serum or Plasma',
       '33762-6_Natriuretic peptide.B prohormone N-Terminal [Mass/volume] in Serum or Plasma',
       '75241-0_Procalcitonin [Mass/volume] in Serum or Plasma by Immunoassay',
       '48058-2_Fibrin D-dimer DDU [Mass/volume] in Platelet poor plasma by Immunoassay',
       '2276-4_Ferritin [Mass/volume] in Serum or Plasma',
       '1988-5_C reactive protein [Mass/volume] in Serum or Plasma',
       '4548-4_Hemoglobin A1c/Hemoglobin.total in Blood',
       '2823-3_Potassium [Moles/volume] in Serum or Plasma',
       '2075-0_Chloride [Moles/volume] in Serum or Plasma',
       '1963-8_Bicarbonate [Moles/volume] in Serum or Plasma',
       '3094-0_Urea nitrogen [Mass/volume] in Serum or Plasma',
       '2160-0_Creatinine [Mass/volume] in Serum or Plasma',
       '62238-1_Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (CKD-EPI)',
       '33254-4_pH of Arterial blood adjusted to patient\'s actual temperature',
       '30341-2_Erythrocyte sedimentation rate',
       '2345-7_Glucose [Mass/volume] in Serum or Plasma',
       '13457-7_Cholesterol in LDL [Mass/volume] in Serum or Plasma by calculation',
       '13458-5_Cholesterol in VLDL [Mass/volume] in Serum or Plasma by calculation',
       '2571-8_Triglyceride [Mass/volume] in Serum or Plasma',
       '2085-9_Cholesterol in HDL [Mass/volume] in Serum or Plasma']

    _pre_known_rename = ["8331-1_Oral temperature",
        "39156-5_Body mass index (BMI) [Ratio]"]

    _pre_known_features = ['age.splits',
        'gender_concept_name',
        'kidney_replacement_therapy',
        'kidney_transplant',
        'htn_v',
        'dm_v',
        'cad_v',
        'hf_ef_v',
        'ckd_v',
        'malignancies_v',
        'copd_v',
        'other_lung_disease_v',
        'acei_v',
        'arb_v',
        'antibiotics_use_v',
        'nsaid_use_v',
        'smoking_status_v',
        'cough_v',
        'dyspnea_admission_v',
        'nausea_v',
        'vomiting_v',
        'diarrhea_v',
        'abdominal_pain_v',
        'fever_v',
        'Oral temperature',
        #'BMI.over30', 'BMI.over35',
        #'76282-3_Heart rate.beat-to-beat by EKG',
        #'8480-6_Systolic blood pressure',
        'Body mass index (BMI) [Ratio]']
    
    pre_known_features_onehot = ['kidney_replacement_therapy', 'kidney_transplant', 'htn_v', 'dm_v',
        'cad_v', 'ckd_v', 'malignancies_v', 'copd_v', 'other_lung_disease_v',
        'acei_v', 'arb_v', 'antibiotics_use_v', 'nsaid_use_v', 'cough_v',
        'dyspnea_admission_v', 'nausea_v', 'vomiting_v', 'diarrhea_v',
        'abdominal_pain_v', 'fever_v', 'Oral temperature',
        'Body mass index (BMI) [Ratio]', 'age.splits_[18,59]',
        'age.splits_[59,74]', 'age.splits_[74,90]',
        'gender_concept_name_FEMALE', 'gender_concept_name_MALE',
        'gender_concept_name_nan', 'hf_ef_v_HFpEF', 'hf_ef_v_HFrEF',
        'hf_ef_v_No', 'hf_ef_v_nan', 'smoking_status_v_Current',
        'smoking_status_v_Former', 'smoking_status_v_Never',
        'smoking_status_v_nan']
    
    lab_feature_onehot = ["Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma by No addition of P-5'-P",
        'Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma',
        'Bicarbonate [Moles/volume] in Serum or Plasma',
        'C reactive protein [Mass/volume] in Serum or Plasma',
        'Chloride [Moles/volume] in Serum or Plasma',
        'Cholesterol in HDL [Mass/volume] in Serum or Plasma',
        'Cholesterol in LDL [Mass/volume] in Serum or Plasma by calculation',
        'Cholesterol in VLDL [Mass/volume] in Serum or Plasma by calculation',
        'Creatine kinase [Enzymatic activity/volume] in Serum or Plasma',
        'Creatinine [Mass/volume] in Serum or Plasma',
        'Erythrocyte sedimentation rate',
        'Ferritin [Mass/volume] in Serum or Plasma',
        'Fibrin D-dimer DDU [Mass/volume] in Platelet poor plasma by Immunoassay',
        'Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (CKD-EPI)',
        'Glucose [Mass/volume] in Serum or Plasma',
        'Heart rate.beat-to-beat by EKG',
        'Hemoglobin A1c/Hemoglobin.total in Blood',
        'Lactate [Moles/volume] in Serum or Plasma',
        'Leukocytes [#/volume] corrected for nucleated erythrocytes in Blood by Automated count',
        'Lymphocytes [#/volume] in Blood by Automated count',
        'Mean blood pressure by Noninvasive',
        'Microscopic_hematuria.above2',
        'Natriuretic peptide.B prohormone N-Terminal [Mass/volume] in Serum or Plasma',
        'Neutrophils [#/volume] in Blood by Automated count',
        'Oxygen saturation in Arterial blood by Pulse oximetry',
        'Potassium [Moles/volume] in Serum or Plasma',
        'Procalcitonin [Mass/volume] in Serum or Plasma by Immunoassay',
        'Proteinuria.above80',
        'Respiratory rate',
        'Sodium [Moles/volume] in Serum or Plasma',
        'Systolic blood pressure',
        'Triglyceride [Mass/volume] in Serum or Plasma',
        'Troponin T.cardiac [Mass/volume] in Serum or Plasma',
        'Urea nitrogen [Mass/volume] in Serum or Plasma',
        'Urine.protein_Abnormal',
        'Urine.protein_Normal',
        'Urine.protein_nan',
        "pH of Arterial blood adjusted to patient's actual temperature"]

    _targets = ["last.status", "is_icu", "was_ventilated"]

    @staticmethod
    def _prettify_column_names(in_names): 
        return_dict = {}
        for s in in_names:
            if re.match("^\d*-\d",s):
                return_dict[s] = s.split("_")[1]
            else:
                return_dict[s] = s
        return return_dict

    def _read_ehd(self, fpath:str):
        self.raw_ehd = pd.read_csv(fpath)
        # they have this typo in the age.splits column where they do '(18,59]' and we fix that
        self.raw_ehd["age.splits"] = self.raw_ehd["age.splits"].str.replace(
            "(", "[", regex=True
        )
        # somehow this is in the originial dataframe twice (hence the .1 at the end of the cname)
        self.raw_ehd = self.raw_ehd.drop("2951-2_Sodium [Moles/volume] in Serum or Plasma.1", axis=1)

        # rename the columns to some prettier way
        column_mapping = LungData._prettify_column_names(self._lab_features_with_codes) # lab features
        self.raw_ehd = self.raw_ehd.rename(columns=column_mapping)
        column_mapping = LungData._prettify_column_names(self._pre_known_rename) # non-lab feats
        self.raw_ehd = self.raw_ehd.rename(columns=column_mapping)


    @staticmethod
    def _get_pretty_label(labelstring:str):
        """
        Return pretty label for labelstring. 
        labelstring of [000, 001, 010, 011, 100, 101, 111], translating to
        last.status + is_icu + was_ventilated
        """
        ret = ""
        if labelstring == "000":
            ret = "Hospitalized (only)"
        elif labelstring == "001":
            ret = "Ventilated"
        elif labelstring == "011":
            ret = "ICU + Vent"
        elif labelstring == "010":
            ret = "ICU"
        elif labelstring == "100":
            ret = "Deceased"
        elif labelstring == "101":
            ret = "Deceased + Ventilated"
        elif labelstring == "110":
            ret = "Deceased + ICU"
        elif labelstring == "111":
            ret = "Deceased + ICU + Ventilated"
        return ret

    def _set_targets(self):
        self.verbose_target_df = self.raw_ehd[self._targets].reset_index(drop=True)
        self.target_df = self.verbose_target_df.replace(
            {"discharged":0, "deceased":1,
             False:0, True:1, "No":0, "Yes":1})
        lbls = self.target_df.astype(int)
        lbls["foo"] = lbls["last.status"].astype(str) + lbls["is_icu"].astype(str) + lbls["was_ventilated"].astype(str)
        le = LabelEncoder()
        y = le.fit_transform(lbls["foo"])
        self.ml_target_df = pd.DataFrame(y, columns=["y"])
        self.ml_target_df["labelstring"] = lbls["foo"]
        _pretty_labels = []
        for i,row in self.ml_target_df.iterrows():
            _pretty_labels.append(LungData._get_pretty_label(row["labelstring"]))
        self.ml_target_df["Outcome"] = _pretty_labels


    def __init__(self, rootdir:str, ehd_fpath:str=_default_path, clusters_fpath=None) -> None:
        self.root = rootdir
        
        try:
            assert(os.path.exists(ehd_fpath))
            assert(os.path.exists(rootdir))
        except AssertionError as e:
            msg = "Error checking for files - make sure external drive is plugged in!"
            print(msg, e)
            raise AssertionError(msg)

        self._read_ehd(ehd_fpath)

        self.patients = []
        self.upatients = [] #unique patients
        self.dates = []
        self.dcm = []
        self.png = []
        self.masks_cleaned = []
        self.masks_l = []
        self.masks_r = []
        # loop over files and save all the info
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                if "1.000000" in subdir:
                    if not file.startswith("._"):
                        # foo[len(bar)+1:].split("/")
                        # ['A670621', '12-24-1900-NA-CHEST AP VIEWONLY-58290', '2.000000-AP-52829']
                        pathsplit = subdir[len(rootdir)+1:].split("/") 
                        if file.endswith(".dcm"):
                            self.patients.append(pathsplit[0]) 
                            self.dates.append(pathsplit[1][:10])
                            self.dcm.append(os.path.join(subdir,file)) 
                        elif file.endswith(".png"):
                            if file.endswith("_cleaned.png"):
                                self.masks_cleaned.append(os.path.join(subdir,file))
                            elif file.endswith("l.png"):
                                self.masks_l.append(os.path.join(subdir,file))
                            elif file.endswith("r.png"):
                                self.masks_r.append(os.path.join(subdir,file))
                            elif file.endswith("1-1.png"):
                                self.png.append(os.path.join(subdir,file))
        self.upatients = list(set(self.patients)) # unique patients
        self.raw_ehd = self.raw_ehd[self.raw_ehd["to_patient_id"].isin(self.upatients)].reset_index(drop=True) #only keep those in unique
        self._set_targets()

        # load clusters
        if clusters_fpath is not None:
            self.patient_clusters = joblib.load(clusters_fpath)

    def as_dataframe(self) -> pd.DataFrame:
        """
        Convinience function to output all info as a DataFrame. Hopefully lets
        me re-use most of the application code :-)
        """
        res = pd.DataFrame(columns=self._types)
        res["patient"] = self.patients
        res["dcm"] = self.dcm
        res["png"] = self.png
        res["mask_cleaned"] = self.masks_cleaned
        res["mask_l"] = self.masks_l
        res["mask_r"] = self.masks_r
        res["date"] = self.dates
        res["delta_date"] = self._infer_reference_date(res)
        return res


    @staticmethod
    def prepare_multiclass_for_radiomics(radiomics_df:pd.DataFrame,
             medical_df:pd.DataFrame, verbose=False):
        targets = ["last.status", "is_icu", "was_ventilated"]
        df = radiomics_df.merge(medical_df[targets + ["to_patient_id"]], left_on="id", right_on="to_patient_id").drop("to_patient_id", axis=1)

        df[targets] = df[targets].replace({"discharged":0, "deceased":1, False:0, True:1, "No":0, "Yes":1})
        df_targets = df[targets]
        lbls = df_targets.astype(int)
        lbls["foo"] = lbls["last.status"].astype(str) + lbls["is_icu"].astype(str) + lbls["was_ventilated"].astype(str)
        le = LabelEncoder()
        y = le.fit_transform(lbls["foo"])
        df["y"] = y
        if verbose:
            print("Labels: ['last.status', 'is_icu', 'was_ventilated'], \n        [0:'000' 1:'001' 2:'010' 3:'011' 4:'100' 5:'101' 6:'110' 7:'111']")
            print(df["y"].value_counts())
        df.drop(targets, axis=1, inplace=True)
        return df

    def _infer_reference_date(self, input):
        """
        Infer the date of the image with the help of the medical data. As per 
        description the field "visit_start_datetime" yields the medical data for
        the patients worst state and thus the medical images should be aligned to
        that field.
        """
        _df = input.copy()
        for p in self.upatients:
            refdate = self.raw_ehd[self.raw_ehd["to_patient_id"] == p]["visit_start_datetime"].values[0]
            lst = self.raw_ehd[self.raw_ehd["to_patient_id"] == p]["length_of_stay"].values[0]
            _df.loc[_df["patient"] == p, "ref"] = refdate
            _df.loc[_df["patient"] == p, "lst"] = lst
        _df["delta"] = pd.to_datetime(_df["date"]) - pd.to_datetime(_df["ref"]) 
        _df["delta"] = _df["delta"].dt.days
        return _df["delta"]

    @staticmethod
    def _extract_firstlevel_features(xray_path:str, mask_path:str,
        patient_id:str, isleft:bool, settings:Dict):
        """
        Get a df row and return radiomics firstorder features
        """
        xray = imread(xray_path, as_gray=True)
        if xray.shape != (512,512): #resize to mask shape if necessary
            xray = Image.fromarray(xray).convert("P")
            xray = torchvision.transforms.functional.resize(xray, (512, 512))
        mask_l = imread(mask_path, as_gray=True)

        # results when working on Image vs converting to numpy are equal, 
        # they convert it in the method themselves
        ray = sitk.GetImageFromArray(xray) 
        #roi = sitk.GetImageFromArray(mask[:,:,1]/255)
        roi = sitk.GetImageFromArray(mask_l)

        feats = rad.firstorder.RadiomicsFirstOrder(ray, roi, **settings)
        res = feats.execute()

        _df = pd.DataFrame.from_dict(res, orient="index").T
        _df["id"] = patient_id
        _df["xray_path"] = xray_path
        _df["left"] = isleft
        return _df  

    @staticmethod
    def _extract_multiple_features(xray_path:str, mask_path:str,
        patient_id:str, isleft:bool, featureClasses:List, settings:Dict):
        """
        Get a df row and return radiomics firstorder features
        """
        xray = imread(xray_path, as_gray=True)
        if xray.shape != (512,512): #resize to mask shape if necessary
            xray = Image.fromarray(xray).convert("P")
            xray = torchvision.transforms.functional.resize(xray, (512, 512))
        mask_l = imread(mask_path, as_gray=True)

        # results when working on Image vs converting to numpy are equal, 
        # they convert it in the method themselves
        ray = sitk.GetImageFromArray(xray) 
        #roi = sitk.GetImageFromArray(mask[:,:,1]/255)
        roi = sitk.GetImageFromArray(mask_l)

        _df = pd.DataFrame()
        for fc in featureClasses:
            feats = fc(ray, roi, **settings)
            res = feats.execute()
            _df = pd.concat([_df, pd.DataFrame.from_dict(res, orient="index").T], axis=1)
            #_df["id"] = patient_id
            #_df["left"] = isleft
        _df["id"] = patient_id
        _df["xray_path"] = xray_path
        _df["left"] = isleft
        return _df  
        
    #@staticmethod
    def extract_radiomics_features(self, image_df:pd.DataFrame=None,
        settings:Dict=None, limit:int=None, featureClasses:List=None,
        combined_rois=False):
        """
        Wrapper for radiomics FE. TODO: add more feature classes like in _extract_firstlevel_features
        """
        if image_df is None:
            image_df = self.as_dataframe()
        if settings is None:
            settings = self._settings
        if limit is None:
            limit = image_df.shape[0]

        print(f"extracting radiomics features for {limit} records... This may take a while!")

        rad_results = pd.DataFrame()

        if combined_rois: # use the full cleaned mask to extract the features in one run instead of per lung
            for index, row in tqdm(image_df[:limit].iterrows(), total=limit):
                try:
                    if featureClasses is None: # only first-order features when list is empty
                        _df = self._extract_firstlevel_features(row.png, row.mask_cleaned,
                                        row.patient, isleft=False, settings=settings)
                        rad_results = pd.concat([rad_results, _df], ignore_index=True)
                    else: # go for multiple features
                        _df = self._extract_multiple_features(row.png, row.mask_cleaned,
                                        row.patient, isleft=False, settings=settings,
                                        featureClasses=featureClasses)
                        rad_results = pd.concat([rad_results, _df], ignore_index=True)
                except Error as e:
                    print(f"ERROR: error when handling {row.png}", e)

        else:
            for index, row in tqdm(image_df[:limit].iterrows(), total=limit):
                try:
                    if featureClasses is None: # only first-order features when list is empty
                        _df = self._extract_firstlevel_features(row.png, row.mask_l,
                                        row.patient, isleft=True, settings=settings)
                        rad_results = pd.concat([rad_results, _df], ignore_index=True)
                        _df = self._extract_firstlevel_features(row.png, row.mask_r,
                                        row.patient, isleft=False, settings=settings)
                        rad_results = pd.concat([rad_results, _df], ignore_index=True)
                    else: # go for multiple features
                        _df = self._extract_multiple_features(row.png, row.mask_l,
                                        row.patient, isleft=True, settings=settings,
                                        featureClasses=featureClasses)
                        rad_results = pd.concat([rad_results, _df], ignore_index=True)
                        _df = self._extract_multiple_features(row.png, row.mask_r,
                                        row.patient, isleft=False, settings=settings,
                                        featureClasses=featureClasses)
                        rad_results = pd.concat([rad_results, _df], ignore_index=True)
                except Error as e:
                    print(f"ERROR: error when handling {row.png}", e)

        return rad_results


    def _infer_column_types(self, df:pd.DataFrame) -> List:
        _df = pd.DataFrame(df.isna().sum())
        null_cols = _df[_df[0]>0].T.columns

        na_dict = {}
        for c in null_cols:
            na_dict[c] = dict(df[c].value_counts())

        binary_columns = []
        numeric_columns = []

        self._binary_cols = binary_columns
        self._numeric_cols = numeric_columns
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

    def _remove_redundant_columns(self, df:pd.DataFrame) -> pd.DataFrame:
        try:
            _df = df.copy()
            for _col in self._drop_categorical:
                if _col in df.columns:
                    _df = _df.drop(_col, axis=1)
            return _df
        except Error as msg:
            print("Error dropping rendundant categorical columns: ", msg)

    def _impute_numeric(self, df:pd.DataFrame, numeric_columns:List, mode:str="nn", mi_val=None,
        k_nearest:int=5, k_distance="uniform", random_state=42):
        """
        Impute numeric columns on a given DataFrame. Working on the DataFrame 
        itself.

        Parameters
        ----------
        df : Input DataFrame

        binary_columns : List columns to impute, naturally received from 
            infer_column_types

        mode : Mode used. Default "mean". More to be done 
            - "mean" : default. Will fill NA's with mean column value.
            - "median" : Fill NA's with median column value
            - "nn" : use k-NN
            - "mi" : missing-indicator (fill zeros for instance)
        """
        #_df = df.copy()

        if mode == "mean":
            avg = df[numeric_columns].mean()
            for c in numeric_columns:
                df[c].fillna(avg[c], inplace=True)
        elif mode == "median":
            avg = df[numeric_columns].median()
            for c in numeric_columns:
                df[c].fillna(avg[c], inplace=True)
        elif mode == "nn":
            # this must be done on scaled data 
            ssc =  StandardScaler()
            df = pd.DataFrame(ssc.fit_transform(df), columns=df.columns)
            df = pd.DataFrame(KNNImputer(n_neighbors=k_nearest,
                weights=k_distance).fit_transform(df), columns=df.columns)
            df = pd.DataFrame(ssc.inverse_transform(df), columns=df.columns)
        elif mode == "mi":
            df[numeric_columns] = df[numeric_columns].fillna(mi_val)
        elif mode == "iterative":
            ssc =  StandardScaler()
            df = pd.DataFrame(ssc.fit_transform(df), columns=df.columns)
            df = pd.DataFrame(IterativeImputer(max_iter=100,
                random_state=random_state).fit_transform(df), columns=df.columns)
            df = pd.DataFrame(ssc.inverse_transform(df), columns=df.columns)

        return df


    def _impute_binary(self, df:pd.DataFrame, binary_columns:List, mode:str="nn", mi_val=None):
        """
        Impute binary columns on a given DataFrame. Working on the DataFrame 
        itself.

        Parameters
        ----------
        df : Input DataFrame

        binary_columns : List columns to impute, naturally received from 
            infer_column_types

        mode : Mode used. Default "mean". More to be done 
            - "mean" : default. Will fill NA's with mean column value.
            - "median" : Fill NA's with median column value
            - "nn" : use k-NN
            - "mi" : missing-indicator (fill zeros for instance)
        """

        if mode == "mean":
            avg = df[binary_columns].mean()
            for c in binary_columns:
                df[c].fillna(avg[c], inplace=True)
        elif mode == "median":
            avg = df[binary_columns].median()
            for c in binary_columns:
                df[c].fillna(avg[c], inplace=True)
        elif mode == "mi":
            df[binary_columns] = df[binary_columns].fillna(mi_val)
        #elif mode == "nn":
        #    pass # already done in numeric
        #return df


    def process_features(self, input_df:pd.DataFrame=None, impute:bool=True, 
            remove_identifiers:bool=True, normalize_dates:bool=True,
            binary_mode:str="nn", numeric_mode:str="nn", k_nearest:int=5, 
            onehot:bool=True, k_distance="uniform", mi_val=0,
            encoder=None, remove_redundant=True,
            round_binary=False, onehot_simplified=False,
            return_df=False):
        """
        EHD preprocessing routine
        """
        if input_df is None:
            _df = self.raw_ehd.copy() # work on a copy
        else:
            _df = input_df.copy()

        #o, n, b = detect_types(_df)
        if remove_identifiers:
            to_drop = ["invasive_vent_days", "length_of_stay",
             "Acute.Kidney.Injury..during.hospitalization.",
             "Acute.Hepatic.Injury..during.hospitalization.",
             "to_patient_id", "covid19_statuses", #"was_ventilated",
             "visit_concept_name", "Other.anticoagulation.therapy",
             ]
            _df.drop(to_drop, axis=1, inplace=True)
        
        if normalize_dates: # dates are 
            _df["visit_start_datetime"] = pd.to_datetime(_df["visit_start_datetime"])-datetime.datetime(1901,1,1)
            _df["visit_start_datetime"] = _df["visit_start_datetime"].dt.days
        else:
            if "visit_start_datetime" in _df.columns:
                _df = _df.drop("visit_start_datetime", axis=1)

        # the patients status is given as discharged/deceased
        if "last.status" in _df.columns:
            _df["last.status"].replace({"discharged":0, "deceased":1}, inplace=True) 

        # fill NaNs that make sense
        if onehot_simplified:
            na_zeros = [#"invasive_vent_days", #days are NaN for patients not ventilated
                    "kidney_replacement_therapy", "kidney_transplant", #either 1 or NaN in the whole dataset - treat as binary
                    ]
        else:
            na_zeros = ["invasive_vent_days", #days are NaN for patients not ventilated
                    "kidney_replacement_therapy", "kidney_transplant", #either 1 or NaN in the whole dataset - treat as binary
                    ]
        for _c in na_zeros:
            if _c in _df.columns: # check if its still there
                _df[_c] = _df[_c].fillna(0) 

        # remove redundant columns (BMI: 28, BMI_over30:False, BMI_over35:False -> BMI: 28)
        if remove_redundant:
            _df = self._remove_redundant_columns(_df)

        # manually estimated colums that can be One-Hot Encoded.
        try:
            assert(encoder in [None, "reuse"])
        except AssertionError as msg:
            print("Error handling encoder, make sure its in [None, \"reuse\"]", msg)

        if onehot:
            if onehot_simplified:
                cat_onehot = ["age.splits", "gender_concept_name", "hf_ef_v",
                                "smoking_status_v"]
            else:
                cat_onehot = ["age.splits", # those do not have missing data entries
                        "gender_concept_name", "Urine.protein", "hf_ef_v", 
                        "smoking_status_v", #"Other.anticoagulation.therapy"
                        ]
            if encoder is None:
                enc = OneHotEncoder().fit(_df[cat_onehot])
                self.pp_enc = enc
            elif encoder == "reuse":
                enc = self.pp_enc
            encoded = pd.DataFrame(enc.transform(_df[cat_onehot]).toarray(),
                    columns=enc.get_feature_names_out())
            _df = pd.concat([_df.drop(columns=cat_onehot).reset_index(drop=True),
                    encoded], axis=1)

        _df.replace({"No":0, "Yes":1,
                    False:0, True:1,
                    }, inplace=True)

        ### Imputation 
        if impute:
            _target_df = None
            # remove targets before any imputation could happen
            # if pd.Series(self._targets).isin(_df.columns).any():
            #     print("REMOVING THEM")
            #     _target_df = _df[self._targets]
            #     _df = _df.drop(self._targets, axis=1)
            
            binary_columns, numeric_columns = self._infer_column_types(_df) # infer column types again
            if numeric_mode == "nn":
                _df = self._impute_numeric(_df, numeric_columns,
                                 mode=numeric_mode, k_nearest=k_nearest,
                                 k_distance=k_distance)
            elif numeric_mode == "del" or binary_mode == "del":
                print("deleting")
                _df = _df.dropna(axis=0)
            else:
                _df = self._impute_numeric(_df, numeric_columns, mode=numeric_mode, mi_val=mi_val)
                self._impute_binary(_df, binary_columns=binary_columns, mode=binary_mode, mi_val=mi_val)
        
        if round_binary: 
            _df[binary_columns] = _df[binary_columns].round() # has_fever: 0.7 -> has_fever: 1
        # if _target_df is not None:
        #     print("ADDING THEM")
        #     _df[self._targets] = _target_df
        if input_df is None:
            self.pp_ehd = _df
        if return_df:
            return _df


    def _define_scale_pipe(self, pca:bool=True, random_state:int=42) -> Pipeline:
        """
        Helper function creating the clustering pipeline
        """

        #pipe = Pipeline([]) #empty pipeline initialization does not work
        if scale:
            pipe=Pipeline([('scaler', StandardScaler())])
        if pca: 
            pipe.steps.append(('pca', TSNE(n_components=2, learning_rate="auto",
                                init="pca", random_state=random_state)))
        return pipe


    def cluster(self, df_input,  method:str="K-means", n_clusters:int=4, scale:bool=True,
        pca:bool=True, n_comp:int=2, tsne:bool=False,
        random_state:int=42): 
        """
        Compute classes using unsupervised Clustering. 
        
        Parameters
        ----------
        df : Input (preprocessed+scaled) DataFrame. 

        n_clusters : Define number of Clusters. Default 5 as per Notebook 
            exploration, yielded most promising results.

        method : Which method used. Default "k-means".  TODO: add more?
            - "k-means" : default. Use k-means.
            - "ward" : ward scaling.

        t-SNE : If t-SNE is used or not.
        
        scale : Use a Standardscaler or not prior to PCA/t-SNE.
        """

        _df = df_input.copy()
        print(method)
        try:
            assert method in ["ward", "K-means"]
        except AssertionError as e:
            print("Method not defined!", e)

        classes = []
        model = None

        if method == "K-means":
            model = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
        elif method == "Ward":
            model = AgglomerativeClustering(linkage="ward", n_clusters=n_clusters)
        # elif method == "DBSCAN":
        #     model = DBSCAN()
        else:
            raise NotImplementedError("No other methods implemented for now.")
        
        if pca: 
            _df = PCA(n_components=n_comp).fit_transform(_df)

        #_X = pipe.fit_transform(_df) # preproc as defined per params
        clusters = model.fit_predict(_df) # predict clusters
        clusters = clusters +1 # start cluster indexing at 1 rather than 0
        #self.classes = classes
        return clusters, _df