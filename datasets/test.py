
import argparse
import torch
import torchvision
from gqa_tweaked import build 
import json


parser = argparse.ArgumentParser(description='Dataset')

parser.add_argument('--vg_img_path',default='/home/aneesh/datasets/gqa_imgs/images/', 
                help='path to image directory')
parser.add_argument('--text_encoder_type',default='roberta-base', 
                help='tokenizer')


parser.add_argument('--do_qa', action="store_true", 
                help="Whether to do question answering")
parser.add_argument('--gqa_ann_path',default='/home/aneesh/datasets/gqa_ann/OpenSource/', 
                help='path to annotations')
parser.add_argument('--gqa_split_type',default='balanced', 
                help='GQA split')
# Segmentation
parser.add_argument(
        "--mask_model",
        default="none",
        type=str,
        choices=("none", "smallconv", "v2"),
        help="Segmentation head to be used (if None, segmentation will not be trained)",
    )
parser.add_argument('--masks',action='store_true') 

args = parser.parse_args()

if args.mask_model != "none":
    args.masks = True
# if args.frozen_weights is not None:
#     assert args.masks, "Frozen training is meant for segmentation only"


args.do_qa = True
dataset = build(image_set='train', 
                args=args)

img, trg = dataset[3]
imset='train'
with open(args.gqa_ann_path + "gqa_answer2id.json", "r") as f:
    answer2id = json.load(f)

id2answer = {i:n for n, i in zip(answer2id.keys(), answer2id.values())}
print(trg['caption'], id2answer[trg['answer'].item()])

print(len(id2answer))
print(answer2id['unknown'])

