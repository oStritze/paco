"""
Module for cleaning the masks as proposed in mask_separator.ipynb

Masks are populated by faulty areas, here we remove small areas and separate
the Left/Right lung from the mask to separate masks for radiomic feature 
extraction.
"""

from PIL import Image
from matplotlib.image import imsave
import numpy as np
from skimage.io import imread
from skimage.draw import polygon
#from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray, rgba2rgb
from skimage import measure


def mask_separator(input, debug=True, save_images=True):

    image_gray = np.zeros((512,512))
    handling_string = False
    print(type(input))
    if isinstance(input, str):
        handling_string = True
        image = imread(input)
        image_gray = rgb2gray(rgba2rgb(image))
        image_gray = (image_gray > image_gray.min()).astype(int) # binarize grayscale

    elif isinstance(input, np.ndarray):
        image = np.array(input)
        image_gray = rgb2gray(image)
        image_gray = (image_gray > image_gray.min()).astype(int) # binarize grayscale

    contours = measure.find_contours(image_gray, 0.0001) # using min() value here had bad influences!
    if debug:
        print(input)
        print("Nr of contours: ", len(contours))
    d = {}
    for i, cont in enumerate(contours):
        # it can happen, that the length of the contours is equal...
        _broken = False
        if len(cont) in d.keys():
            cnt = 1
            _broken = True
            while _broken:
                if len(cont)+cnt in d.keys():
                    cnt+=1 # recursively add 1 until we find a free dict, hope this wont matter too much
                d[len(cont)+cnt] = i # just add a 1...
                _broken = False
        else:
            d[len(cont)] = i

    biggest = sorted(d.keys())[-2:] # take two biggest contours -> with most points
    biggest_c = [d[i] for i in biggest]

    d = {}
    for cnt, i in enumerate(biggest_c):
        cont = contours[i]
        mask = np.zeros(image_gray.shape)
        rr, cc = polygon(cont[:, 0], cont[:, 1], mask.shape)
        mask[rr, cc] = 1
        d[cnt] = {"mask": mask,
        "center_x": cont[:, 1].mean(), "center_y": cont[:, 0].mean()}

    if debug:
        print(input)
        print("Nr of elements in dict: ", len(d))

    try:
        if d[0]["center_x"] > d[1]["center_x"]:
            mask_l = d[0]["mask"] # left lung is the right one in the image
            mask_r = d[1]["mask"] 
        else:
            mask_r = d[0]["mask"] 
            mask_l = d[1]["mask"] 
    except Exception as e:
        #print(len(d))
        #print(e)
        if handling_string:
            print(f"Error processing {input}! Check check")
        print(d[0]["center_x"], d[1]["center_x"])

    mask_combined = mask_r + mask_l
    # in case of overlapping masks, it occured that the max value is greater 1 and thus the mask needs to be binarized again
    mask_combined = (mask_combined > mask_combined.min()).astype(int)
    

    # fpath= "/Volumes/Samsung_T5/MA/manifest-1638522923319/subsample/A700420/01-28-1901-NA-CHEST AP VIEWONLY-59525/1.000000-AP-49664/1-1_mask.png"
    ## remove .png and put _L / _R.png when saving
    if save_images and handling_string:
        imsave(input[:-4]+"_r.png", mask_r, cmap="gray")
        imsave(input[:-4]+"_l.png", mask_l, cmap="gray")
        imsave(input[:-4]+"_cleaned.png", mask_combined, cmap="gray")
    elif not save_images:
        return mask_combined, mask_r, mask_l
