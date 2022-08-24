from typing import Dict
import pandas as pd
import numpy as np
import os
import random
import radiomics as rad
import pydicom
from PIL import Image, ImageOps
import torchvision
import SimpleITK as sitk
import radiomics as rad
from skimage.io import imread


def get_xray_paths(rootdir:str=None):
    """
    Get paths for xrays based on rootdir. 
    """
    if rootdir is None:
        rootdir = f"/Volumes/Samsung_T5/MA/manifest-1638522923319/subsample/"

    xrays = []

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if not file.startswith("._"):
                if file != ".DS_Store":
                    xrays.append(os.path.join(subdir, file))
    return xrays

def _infer_reference_date(input:pd.DataFrame, medical_df:pd.DataFrame()):
    """
    Infer the date of the image with the help of the medical data. As per 
    description the field "visit_start_datetime" yields the medical data for
    the patients worst state and thus the medical images should be aligned to
    that field.
    """
    _df = input.copy()
    # for each patient loop through input and get the reference date
    for patient in _df["patient"].unique():
        refdate = medical_df[medical_df["to_patient_id"] == patient]["visit_start_datetime"].values[0]
        lst = medical_df[medical_df["to_patient_id"] == patient]["length_of_stay"].values[0]
        _df.loc[_df["patient"] == patient, "ref"] = refdate
        _df.loc[_df["patient"] == patient, "lst"] = lst

    _df["delta"] = pd.to_datetime(_df["date"]) - pd.to_datetime(_df["ref"]) 
    _df["delta"] = _df["delta"].dt.days
    #_df.to_csv("data/foo.csv", index=False)
    return _df

def get_segmentation_paths_df(subsample=True, local_sample=False,
    rootdir:str=None, return_masks=True, medical_df:pd.DataFrame=None, dedub=True):
    """
    Get paths for the images and segmentations and patients and return it as a
    pandas DF.

    Parameters
    ----------
    subsample : Use a smaller subsample for development purposes. Default=True

    local_sample : TODO: Not implemented -- support local data in workdir/data/* 
        directory for instance.
    """
    res = pd.DataFrame(columns=["patient", "dcm", "mask", "date"])

    path = ""
    patients = []
    dcm = []
    masks = []
    dates = []
    if rootdir is None:
        if not local_sample and subsample:
            path = "/Volumes/Samsung_T5/MA/manifest-1638522923319/subsample"
    elif rootdir is not None:
        path = rootdir
        
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if not file.startswith("._"):
                # foo[len(bar)+1:].split("/")
                # ['A670621', '12-24-1900-NA-CHEST AP VIEWONLY-58290', '2.000000-AP-52829']
                pathsplit = subdir[len(path)+1:].split("/") 
                if file.endswith(".dcm"):
                    patients.append(pathsplit[0]) 
                    dates.append(pathsplit[1][:10])
                    dcm.append(os.path.join(subdir,file)) 
                elif file.endswith(".png"):
                    masks.append(os.path.join(subdir,file)) 

    res["patient"] = patients
    res["dcm"] = dcm
    res["mask"] = masks
    res["date"] = dates
    res = _infer_reference_date(res, medical_df)
    
    if dedub:
        res = res[res["dcm"].str.contains("1.000000")]
    return res


def get_segmentation_paths(subsample=True, local_sample=False, rootdir:str=None, return_masks=True):
    """
    Get paths for the images and segmentations and patients.

    Parameters
    ----------
    subsample : Use a smaller subsample for development purposes. Default=True

    local_sample : TODO: Not implemented -- support local data in workdir/data/* 
        directory for instance.
    """
    path = ""
    patients = []
    dcm = []
    masks = []
    if rootdir is None:
        if not local_sample and subsample:
            path = "/Volumes/Samsung_T5/MA/manifest-1638522923319/subsample"
    elif rootdir is not None:
        path = rootdir
        
    for p in os.listdir(path):
        if not p.startswith("._"):
            patients.append(p)

    for subdir, dirs, files in os.walk(path):
        for file in files:
            if not file.startswith("._"):
                if file.endswith(".dcm"):
                    dcm.append(os.path.join(subdir,file))
                elif file.endswith(".png"):
                    masks.append(os.path.join(subdir,file))

    if return_masks:
        return patients, dcm, masks
    else:
        return patients, dcm

def sel_random_patient(patients, dcm):
    """
    Select a random patient from provided lists.
    """
    patient = random.choice(patients) # select random patient
    subselect = []
    for i in dcm: #get all available images for that patient, save in subselect
        if patient in i:
            subselect.append(i) 
    dcm_image = random.choice(subselect) 
    mask_image = dcm_image[:-4]+"_mask.png" 
    # we were returning the whole path before, but build it with appending the
    # /Volumes/... info in the app so only return the image subpath to make it
    # work wich is [64:]
    return patient, dcm_image[64:], mask_image[64:]

def retrieve_image(patient, image, overlay=True,
                path="/Volumes/Samsung_T5/MA/manifest-1638522923319/",
                subsample=True):
    """
    Load image from string information.
    """
    if image.startswith(path):
        path = image
    else: # when using the new df with paths in it we do not need this
        if subsample:
            path = path + "subsample/"
        else:
            path = path + "COVID-19-NY-SBU/"
        path = f"{path}{patient}/{image}" 

    if path.endswith(".dcm"):
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array
        img = np.round(img/img.max()*255) #scale between [0,255]
        img = Image.fromarray(img).convert("P")
        # resize to 512x512
        img = torchvision.transforms.functional.resize(img, (512, 512))
    elif path.endswith(".png"):
        img = Image.open(path).convert("P")
        # convert this to a green channel picture only - if we want the grayscale we can use it just like that
        # if we want more (for instance for plotting the mask) we can use it as well
        img = np.stack(( np.zeros((512,512)), np.asarray(img), np.zeros((512,512)) ), axis=2)
        #img = img*255
        if overlay:
            dcmpath = path[:-9] + ".dcm"
            dcm = pydicom.dcmread(dcmpath)
            background = dcm.pixel_array
            background = np.round(background/background.max()*255) #scale between [0,255]
            background = Image.fromarray(background).convert("P")
            background = torchvision.transforms.functional.resize(background, (512, 512))

            img = np.array(img*255, dtype=np.uint8) # for Image.blend() we need to convert to uint8
            img = Image.blend(background.convert("RGB"), Image.fromarray(img), 0.2)
            #img = Image.blend(img.convert("RGB"), background, 0.2)
    return img

def retrieve_image_from_path(path:str, overlay=False):
    img = np.array((512,512))
    return img


def get_pretty_labels():
    pretty_labels = {
        "last.status": "Deceased",
        "is_icu": "ICU Treatment",
        "was_ventilated": "Ventilated",
        "length_of_stay": "Length of stay",
    }

    return pretty_labels

def extract_firstlevel_features(xray_path:str, mask_path:str, patient_id:str, isleft:bool, settings:Dict):
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
    _df["left"] = isleft

    return _df