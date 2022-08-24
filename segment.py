"""
    Main segmentation routine. Create segmentation masks from transfer-learning 
    network. 

    Masks are also separated and saved to respective files if __clean__ is set to 
    True.

    Running examples below.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from src.segmentation import segmentation
from src.segmentation.models import PretrainedUNet, tuned_PretrainedUNet
import torch
from tqdm import tqdm
from src.segmentation.mask_cleaning import mask_separator
import time
import argparse
torch.manual_seed(42) # set seed for reproducibility!

def main(args):
    
    try:
        assert(args.preproc in ["None", "blur", "median"])
    except AssertionError as e:
        print(e)

    _preproc = None

    if args.preproc != "None":
        if args.preproc == "blur":
            _preproc = segmentation.blur_adapthisteq
        elif args.preproc == "median":
            _preproc = segmentation.median_adapthisteq

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Torch device: ", device)
    

    ## Load pre-learnt
    model_name = args.model
    print(f"Loading pre-trained model: {model_name}...")
    
    ## Prepare U-Net
    if model_name.startswith("t_"):
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

    unet.load_state_dict(torch.load(f"models/lung_seg/{model_name}", map_location=torch.device("cpu")))
    unet.to(device)

    ## get list of images to segment
    rootdir = args.rootdir
    xrays = []

    if not args.ignore_struct:
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                if "1.000000" in subdir: # only those files
                    if not file.startswith("._"): # weird file-locks on the external drive
                        if file.endswith(".dcm"): # ignore other files like previous masks and lovely .DS_Store
                            xrays.append(os.path.join(subdir, file))

    elif args.ignore_struct:
        print("ignoring structure...")
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                if not file.startswith("._"): # weird file-locks on the external drive
                    if file.endswith(".dcm"): # ignore other files like previous masks and lovely .DS_Store
                        xrays.append(os.path.join(subdir, file))

    total_count = len(xrays)
    batch_size = args.batch
    chunks = (total_count - 1) // batch_size + 1
    print(f"Processing a totel of {total_count} images in {chunks} batches, using preproc strategy {_preproc}...")
    start_time = time.time()
    for i in tqdm(range(chunks)):
        batch = xrays[i*batch_size:(i+1)*batch_size]

        # segment
        in_images, out_images = segmentation.process_dcim(batch, unet, device, _preproc, args.save_png)
        # save masks
        for path, img in tqdm(zip(batch, out_images)):
            fname = path[:-4] + "_mask.png"
            #print(fname)
            plt.imsave(fname, img, cmap="gray")
            
            if args.clean: # clean and seperate masks on the fly
                mask_separator(fname)
    print("--- DONE! Took {}s".format(round(time.time()-start_time), 2))

if __name__ == "__main__":
    """
    debug:
    python3 segment.py --batch 16 --rootdir /Volumes/Samsung_T5/MA/manifest-1641816674790/testing/ --save_png True --model unet-6v.pt --preproc blur
    
    full: 
    python3 segment.py

    reverse accuracy:
    python3 segment.py --batch 5 --rootdir /Volumes/Samsung_T5/MA/manifest-1641816674790/revacc_0/ --save_png True --ignore_struct True
    python3 segment.py --batch 5 --rootdir /Volumes/Samsung_T5/MA/manifest-1641816674790/revacc_1/ --save_png True --ignore_struct True --model t_unet_v4_adam_es20pct_rot25.pt
    python3 segment.py --batch 5 --rootdir /Volumes/Samsung_T5/MA/manifest-1641816674790/revacc_2/ --save_png True --ignore_struct True --preproc blur
    python3 segment.py --batch 5 --rootdir /Volumes/Samsung_T5/MA/manifest-1641816674790/revacc_3/ --save_png True --ignore_struct True --preproc blur --model t_unet_v4_adam_es20pct_rot25.pt
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", type=str,
     default="/Volumes/Samsung_T5/MA/manifest-1641816674790/subsample_thresh2/")
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--model", default="unet-6v.pt")
    parser.add_argument("--clean", type=bool, default=True)
    parser.add_argument("--preproc", type=str, default="None")
    parser.add_argument("--save_png", type=bool, default=False)
    parser.add_argument("--ignore_struct", type=bool, default=False)
    args = parser.parse_args()

    main(args)