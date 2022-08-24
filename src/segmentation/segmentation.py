from pydicom import dcmread
import torch, torchvision
from PIL import Image, ImageOps
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.filters import median, gaussian
from skimage.exposure import equalize_adapthist, equalize_hist

torch.manual_seed(42)

def blur_adapthisteq(img, sigma=1, clip_limit=0.01):
    """
    Blur + adaptive histogram equalization on an image for contrast enhancement.
    """
    _img = img.copy()
    _img = gaussian(_img, sigma=sigma)
    _img = equalize_adapthist(_img, clip_limit=clip_limit)
    return _img

def median_adapthisteq(img, disk_size = 3, clip_limit=0.0075):
    """
    Median + adaptive histogram equal. for contrast enhancement. 
    Takes twice as long as blur!
    """
    _img = img.copy()
    _img = median(_img, disk(disk_size))
    _img = equalize_adapthist(_img, clip_limit=clip_limit)
    return _img


def process_dcim(filepaths, model, device, preproc=None, save_png=False, verbose=False):
    """
    Get a list of dcm-images and return the original and segmented lung mask
    
    Usage:
    ```
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    unet = PretrainedUNet(
        in_channels=1,
        out_channels=2, 
        batch_norm=True, 
        upscale_mode="bilinear"
    )
    model_name = "unet-6v.pt"
    unet.load_state_dict(torch.load(f"models/{model_name}", map_location=torch.device("cpu")))
    unet.to(device)
    process_dcim(xrays, unet, device)
    ```
    """
    input = []
    output = []
    
    for fpath in filepaths:

        dcm = dcmread(fpath)
        origin = dcm.pixel_array

        if preproc is not None:
            origin = preproc(origin)
            origin = np.round(origin * 255)
            origin = Image.fromarray(origin).convert("P")
        else:
            origin = np.round(origin/origin.max() * 255) 
            origin = Image.fromarray(origin).convert("P")
        #origin = np.round(origin/origin.max()*255) #scale between [0,1]
        #origin = Image.fromarray(origin).convert("P")
        
        if save_png:
            #print(fpath[:-3]+"png")
            origin.save(fpath[:-3]+"png")
        
        origin = torchvision.transforms.functional.resize(origin, (512, 512))
        origin = torchvision.transforms.functional.to_tensor(origin) - 0.5

        input.append(origin)
    
    cnt = range(len(input))
    for origin, c in zip(input, cnt):
        if verbose:
            print(f"processing {filepaths[c]}... ")
        with torch.no_grad():
            origin = torch.stack([origin])
            origin = origin.to(device)
            out = model(origin)
            softmax = torch.nn.functional.log_softmax(out, dim=1)
            out = torch.argmax(softmax, dim=1)
            
            origin = origin[0].to("cpu")
            out = out[0].to("cpu")
        output.append(out)

    assert(len(input) == len(output))

    return input, output

def plot_mask(xray, mask):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("origin image")
    plt.imshow(xray)

    plt.subplot(1, 3, 2)
    plt.title("blended origin")
    plt.imshow(Image.blend(xray.convert("RGB"), Image.fromarray(mask), 0.2))

    plt.subplot(1, 3, 3)
    plt.title("mask")
    plt.imshow(mask)