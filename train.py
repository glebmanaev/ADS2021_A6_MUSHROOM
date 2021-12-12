from detectron2.utils.logger import setup_logger
import numpy as np
import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances


setup_logger()
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml") #Remove if you want training initialization from random weights
cfg.SOLVER.MAX_ITER = 210000
register_coco_instances("mushrooms", {}, "data/train.json", "data/mushroom_train")
cfg.DATASETS.TRAIN = ('mushrooms',)
cfg.DATASETS.TEST = ()

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
