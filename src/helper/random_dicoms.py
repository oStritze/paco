import os
import random
import argparse
import errno, shutil


def copy(src, dest):
    try:
        shutil.copytree(src, dest)
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            #os.umask(0)
            #print("trying to make dirs at: ", dest.split("1-1.dcm")[0])
            os.makedirs(dest.split("1-1.dcm")[0], exist_ok=True)
            shutil.copy(src, dest)
        else:
            print('Directory not copied. Error: %s' % e)


def sel_random(args):
    path = args.rootdir
    random.seed(args.seed)
    dcms = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
                if not file.startswith("._"):
                    if file.endswith(".dcm"):
                        if "1.000000" in subdir:
                            dcms.append(os.path.join(subdir,file)) 

    prct = args.perc
    n = round(len(dcms)/100*prct) # select 1% of the whole dataset
    selection = random.choices(dcms, k=n)

    print(f"copying {n} dicoms...")

    for s in selection:
        fs = s.split("COVID-19-NY-SBU/")[1]
        fs = fs.split("1.000000-AP-")[1]
        #print(s)
        #print(f"{args.dest}{fs}")
        copy(f"{s}", f"{args.dest}{fs}")

    print(f"--- DONE!")

def structure(args):
    root = args.rootdir
    dest = args.rootdir

    if not os.path.exists(dest+"masks"):
        os.makedirs(dest+"masks")
    if not os.path.exists(dest+"images"):
        os.makedirs(dest+"images")

    for dir in os.listdir(root):
        if dir not in ["images", "masks"]:
            mask = root+dir+"/1-1_mask_cleaned.png"
            #print(mask)
            #print(root+"masks/"+dir+".png")
            img = root+dir+"/1-1.png"
            shutil.copy(mask, root+"masks/"+dir+".png")
            shutil.copy(img, root+"images/"+dir+".png")
    

if __name__ == "__main__":
    """
    --copy:
    python3 src/helper/random_dicoms.py --dest /Volumes/Samsung_T5/MA/manifest-1641816674790/revacc_1/
    --struct:
    python3 src/helper/random_dicoms.py --mode struct --rootdir /Volumes/Samsung_T5/MA/manifest-1641816674790/revacc_0/ 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="copy")
    parser.add_argument("--rootdir", type=str,
        default="/Volumes/Samsung_T5/MA/manifest-1641816674790/COVID-19-NY-SBU/")
    parser.add_argument("--dest", type=str,
        default="/Volumes/Samsung_T5/MA/manifest-1641816674790/revacc_0/")
    parser.add_argument("--perc", type=int,
        default=1)
    parser.add_argument("--seed", type=int,
        default=42)
    args = parser.parse_args()

    print("Mode: ", args.mode)
    if args.mode == "copy":
        sel_random(args)
    elif args.mode == "struct":
        structure(args)
