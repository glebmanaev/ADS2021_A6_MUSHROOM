# Mushroom Project (ADS2021 A6) 

## Team Members:
* Dmytro Fedorenko
* Glib Manaiev

## Installation
1. `git clone https://github.com/glebmanaev/ADS2021_A6_MUSHROOM.git`
2. Install detectron2 following [these instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).
3. Install requierements: ` pip install -r requiremnts.txt `.
4. Install the [dataset](https://www.kaggle.com/maysee/mushrooms-classification-common-genuss-images). Put `Mushrooms` directory to `/data`. *You do not need it for inference.*
5. Download the pretrained model to `/model` from [here](https://drive.google.com/drive/folders/1wkJpA0YfuDeoKePOqhilybwnhYTO8Bcl?usp=sharing).
## Dataset conversion to COCO format
To convert kaggle dataset to COCO format datset run:

    cd data/
    python coco_converter.py --eval_pct 10
    
You can change the percentage of the data used for evaluation set by `--eval_pct`.
## Inference
To test model run:

    python inf_val.py --image_path images_for_inference/test_Hygrocybe_3500

You can change path to the model using `--model /path/to/model`.
## Evaluation
To evaluate model run:

    python inf_val.py --eval True
    
## Training
To train model run:

    python train.py
    
To train model with training data augmentation run:

    python train_data_aug.py
