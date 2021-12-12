import numpy as np
import cv2
import argparse
import json
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances


def get_parser():
    parser = argparse.ArgumentParser(description="InfEva")
    parser.add_argument('-i', '--image_path', help="path to the image",
                        default="images_for_inferece/test_Hygrocybe_3500.jpg")
    parser.add_argument('-m', '--model', help="weights path",
                        default="models/model.pth")
    parser.add_argument('-e', '--eval', type=bool, help="do evaluation instead of inference",
                        default=False)
    return parser


def init_predictor(model='models/model.pth'):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
    cfg.MODEL.WEIGHTS = model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0
    register_coco_instances("mushrooms", {}, "data/train.json", "data/mushroom_train")
    cfg.DATASETS.TRAIN = ('mushrooms',)
    cfg.DATASETS.TEST = ()

    return DefaultPredictor(cfg)


def evaluation(model,annotations):
    results = []
    predictor = init_predictor(model)

    with open(annotations, 'r') as f:
        annos = json.load(f)

    for ann in tqdm(annos['images']):
        im = cv2.imread(f"data/mushroom_eval/{ann['file_name']}")
        gt = ann["category_id"] - 1
        outputs = predictor(im)
        instances = outputs["instances"].to("cpu")
        prediction = instances.pred_classes.numpy()[instances.scores.numpy().argmax()]
        results.append(gt == prediction)
    
    score = int(round(sum(results) / len(results), 2) * 100)
    print(f'Correct guess in {score}% cases')


def inference(model, image_path):
    classes = {0 : "Agaricus",
              1 : "Amanita",
              2 : "Boletus",
              3 : "Cortinarius",
              4 : "Entoloma",
              5 : "Hygrocybe",
              6 : "Lactarius",
              7 : "Russula",
              8 : "Suillus"
              }

    im = cv2.imread(image_path)

    predictor = init_predictor(model)
    outputs = predictor(im)
    instances = outputs["instances"].to("cpu")
    mushroom = instances.pred_classes.numpy()[instances.scores.numpy().argmax()]

    cv2.imshow(classes[mushroom],im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    args = get_parser().parse_args()

    if args.eval:
        evaluation(args.model, 'data/eval.json')
    else:
        inference(args.model, args.image_path)

    

if __name__ == "__main__":
    main()
