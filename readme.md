# PACO
Prediction, Analysis and Communication of COVID-19 hospitalization cases.

We used openly available data from 
[cancer imaging archive](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=89096912), segmenting lungs in X-Rays, extracting radiomcis features and predicting hospitalization outcomes. This was merged with electronic health data available in the dataset as CSV. 


## Run the application

1. Create conda environment from file
    ```
    conda env create --name paco --file=env_nobuilds.yml
    ```
2. Download data X-Ray data from [cancer imaging archive](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=89096912) 
3. Set the __rootdir__ variable in ```app.py``` accordingly to your filepath
4. Download the pre-trained segmentation [PyTorch Networks](https://drive.google.com/drive/folders/1d7ZgQWxOMeLLiJWUPw8Ty6uDWamNWJVZ?usp=sharing)
5. Create segmentations using ```segment.py```
6. Run streamlit app
    ```
    streamlit run app.py
    ```

## Segmentation
Training segmentation models was done in [another repository](https://github.com/oStritze/lung-segmentation). Two models are provided for download here following work of [TernausNet](https://arxiv.org/abs/1801.05746), [Towards Robust Lung Segmentation in Chest Radiographs with Deep Learning](https://arxiv.org/pdf/1811.12638v1.pdf), and the implementation publicly available by [Illia Ovcharenko](https://github.com/IlliaOvcharenko/lung-segmentation).

[PyTorch Download Link](https://drive.google.com/drive/folders/1d7ZgQWxOMeLLiJWUPw8Ty6uDWamNWJVZ?usp=sharing)

## Folder Structure

```src/```: source code modules 

```src/streamlit_tabs```: streamlit tabs loaded by ```app.py```

```models/```: place for storing models that are trained and re-used later on without re-training

```data/```: data folder with ehd and extracted radiomics features. Cleaned radiomics versions are _stacked_ versions where features where highly correlated features were removed (threshhold 0.95).

```series-data1637230308969.csv```: list of patients who have more than 3 images in the dataset (some smaller subselection must be still made)

## Cite

Please cite our work if it helped your research:

```
PLACEHOLDER
```