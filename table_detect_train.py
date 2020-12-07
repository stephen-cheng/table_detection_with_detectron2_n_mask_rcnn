import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import os
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

# if your dataset is in COCO format, this cell can be replaced by the following three lines:
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import print_instances_class_histogram


# dataset setting
register_coco_instances("train_table_1", {}, "dataset/coco/output.json",
                        'dataset/images')
metadata = MetadataCatalog.get("train_table_1")
dataset_dicts = DatasetCatalog.get("train_table_1")

print(print_instances_class_histogram(dataset_dicts , ['table']))

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train_table_1",)
cfg.DATASETS.TEST = ()

# Let training initialize from model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml") 
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.DATALOADER.NUM_WORKERS = 1
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.WARMUP_ITERS = 150
cfg.SOLVER.MAX_ITER = 1000
cfg.SOLVER.STEPS = (1000, 1500)
cfg.INPUT.MIN_SIZE_TRAIN = (100,640, 672, 704, 736, 768)
cfg.TEST.MIN_SIZES = (100,640, 672, 704, 736, 768)
cfg.MODEL.SIZES =  [[16],[32], [64], [128], [256]]
#cfg.SOLVER.GAMMA = 0.05
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
# cfg.OUTPUT_DIR = 'output'
cfg.MODEL.DEVICE = 'cpu'
cfg.TEST.EVAL_PERIOD = 80
cfg.TEST.PRECISE_BN.ENABLED = True

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
