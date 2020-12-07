import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import os
import random
import time
import pandas as pd

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.catalog import Metadata
from detectron2.utils.visualizer import Visualizer


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = 'output/model_final.pth'
cfg.MODEL.DEVICE = 'cpu'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6 # set threshold for this model
predictor = DefaultPredictor(cfg)
metadata = Metadata()
metadata.set(thing_classes=['logo', 'table', 'stamp', 'signature'])

# testing 
start_time=time.time()
path = 'dataset/test'
print('the number of image for prediction is : ' + str(len(os.listdir(path))) )

if not os.path.exists("result"):
    os.makedirs("result")
index=0
for idx , image in enumerate(os.listdir('dataset/test')):
    index = index + 1
    im = cv2.imread(os.path.join(path , image))
    im1 = im.copy()
    outputs= predictor(im)
    v = Visualizer(im[:, :, ::-1], metadata = metadata, scale=0.8)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    pred_class = outputs["instances"].pred_classes.cpu().numpy()
    pred_annot= outputs["instances"].pred_boxes.tensor.cpu().numpy()
    full_pred  = np.concatenate((pred_annot,pred_class.reshape(-1,1)), axis = 1)
    full_annot_df = pd.DataFrame(data = full_pred.astype('int') ,columns=['x1','y1', 'x2', 'y2','class'])
    for idx1,item in enumerate(full_annot_df.iterrows()):
        img1 = im1[item[1]['y1']:item[1]['y2'],item[1]['x1']:item[1]['x2']]
        print(os.path.join(os.path.dirname(path),'line_crop',str(idx)+'_'+str(idx1)+'.png'))
        cv2.imwrite("result/line_crop_" + str(idx)+'_'+str(idx1)+'.png', img1)
    cv2.imwrite("result/img_pred_" + str(index)+".png" , v.get_image()[:, :, ::-1])
end_time=time.time()
print('time for {}'.format(end_time-start_time))


