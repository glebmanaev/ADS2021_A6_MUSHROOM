import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from skimage import io
import copy


def get_parser():
    parser = argparse.ArgumentParser(description="Bebra Converter")
    parser.add_argument('-e', '--eval_pct', type=int, help="eval dataset split percentage (int)", default=1)
    return parser

args = get_parser().parse_args()
ratio = int(100 / args.eval_pct)

folder = ['mushroom_train/','mushroom_eval/']

categories = {"Agaricus": 1,
              "Amanita": 2,
              "Boletus": 3,
              "Cortinarius": 4,
              "Entoloma": 5,
              "Hygrocybe": 6,
              "Lactarius": 7,
              "Russula": 8,
              "Suillus": 9
              }

annos = {}
annos['categories'] = [
    {"supercategory": "mushroom", "id": 1, "name": "Agaricus"},
    {"supercategory": "mushroom", "id": 2, "name": "Amanita"},
    {"supercategory": "mushroom", "id": 3, "name": "Boletus"},
    {"supercategory": "mushroom", "id": 4, "name": "Cortinarius"},
    {"supercategory": "mushroom", "id": 5, "name": "Entoloma"},
    {"supercategory": "mushroom", "id": 6, "name": "Hygrocybe"},
    {"supercategory": "mushroom", "id": 7, "name": "Lactarius"},
    {"supercategory": "mushroom", "id": 8, "name": "Russula"},
    {"supercategory": "mushroom", "id": 9, "name": "Suillus"},
]
annos['images'] = []
annos['annotations'] = []
annos = [annos, copy.deepcopy(annos)]

ids = 0
for mushr_type in tqdm(os.listdir('Mushrooms'), leave=False, desc='Processing mushroom types'):
    for mushroom in tqdm(os.listdir('Mushrooms/'+mushr_type), leave=False, desc='Processing individual mushrooms'):

        try:
            img = io.imread(f'Mushrooms/{mushr_type}/{mushroom}')
        except:
            continue #skipping corrupted images
        img = cv2.imread(f'Mushrooms/{mushr_type}/{mushroom}')
        h, w, _ = img.shape
        ids += 1
        
        if ids % ratio == 0:
            name,ext = os.path.splitext(mushroom)
            new_img_name = f"test_{mushr_type}_{ids}{ext}"
        else:
            name,ext = os.path.splitext(mushroom)
            new_img_name = f"{mushr_type}_{ids}{ext}"

        annos[ids % ratio == 0]['images'].append({
            "file_name": new_img_name,
            "height": h,
            "width": w,
            "category_id": categories[mushr_type],
            "id": ids
        })

        annos[ids % ratio == 0]['annotations'].append({
            "area": h*w,
            "iscrowd": 0,
            "image_id": ids,
            "bbox": [0, 0, h, w],
            "category_id": categories[mushr_type],
            "id": ids
        })

        cv2.imwrite(folder[ids % ratio == 0]+new_img_name,img)

with open('train.json', 'w') as f:
    json.dump(annos[0], f)

with open('test.json', 'w') as f:
    json.dump(annos[1], f)

print('Conversion Completed Successfully')
