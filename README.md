## Install required packages:

```
pip install -r requirements.txt
```

## Install Detectron2

```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```


## Create custom COCO dataset

Then you can run the `voc2coco.py` script to generate a COCO data formatted JSON file for you.
```
python voc2coco.py ./dataset/annotations ./dataset/coco/output.json
```

Further instruction on how to create your own datasets, read the [THIS](https://www.dlology.com/blog/how-to-create-custom-coco-data-set-for-object-detection/).


Then you can run the following Jupyter notebook to visualize the coco annotations. 

`COCO_Image_Viewer.ipynb`

## Training

```
python table_detect_train.py
```

## Evaluation

```
python table_detect_test.py
```


