import torch
import torchvision
import os
from PIL import Image, ImageOps

import pandas as pd
import numpy as np

from segmentation.models import UNet, PretrainedUNet

from pydicom import dcmread

def _prepare_processing():
    """
    Prepare the processing by creating a unet (loading pre-trained) and 
    creating the device for pytorch.
    """

def process_dcim(filepaths, save_inplace=False, device:torch.device=None, unet:torch.nn.Module=None):
    """
    Get a list of dcm-images and return the original and segmented lung mask as arrays.
    TODO: Process directly in here and implement batching for quicker processing. For now it takes ~3sec per image
    """

    if device is None and unet is None:
        device, unet = _prepare_processing()

    input = []
    output = []
    
    for fpath in filepaths:
        dcm = dcmread(fpath)

        origin = dcm.pixel_array
        origin = np.round(origin/origin.max()*255) #scale between [0,1]
        origin = Image.fromarray(origin).convert("P")

        origin = torchvision.transforms.functional.resize(origin, (512, 512))
        origin = torchvision.transforms.functional.to_tensor(origin) - 0.5

        input.append(origin)
    
    cnt = range(len(input))
    for origin, c in zip(input, cnt):
        print(f"processing {filepaths[c]}... ")
        with torch.no_grad():
            origin = torch.stack([origin])
            origin = origin.to(device)
            out = unet(origin)
            softmax = torch.nn.functional.log_softmax(out, dim=1)
            out = torch.argmax(softmax, dim=1)
            
            origin = origin[0].to("cpu")
            out = out[0].to("cpu")
        output.append(out)

    assert(len(input) == len(output))

    return input, output
