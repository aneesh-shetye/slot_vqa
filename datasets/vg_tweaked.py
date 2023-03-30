import json
from pathlib import Path 

import pandas as pd 
import cv2

import numpy as np
from skimage import draw 

import torch
from torchvision import transforms as T

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import CLIPTokenizer



class VGDataset(Dataset): 

    def __init__(self, 
                 root, 
                 resolution, 
                 transform): 
        
        super().__init__()
        self.root = root

        f = open(root + "region_description.json")
        self.reg_descriptions = json.load(f)

        self.T = transform
        self.res = resolution        
        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')


    def __len__(self): 
        return len(self.reg_descriptions['regions'])
         

    def __getitem__(self, index: int): 

        img_id = self.reg_descriptions[index]['regions'][index]['image_id']
        img = cv2.imread(f'{self.root}/images/{img_id}.jpg')
        img = torch.tensor(img)
        img = img.permute(-1, 0, 1)

        phrases = []
        for i in self.reg_descriptions[index]['regions'][index]: 
            phrases.append(i['phrase'].lower(), max_length=512, truncation=True, padding=True)['input_ids'] 

        phrases = self.tokenizer(phrases)

        resize = T.Resize(self.res)
        img = resize(img) 

        for t in self.T:
            img = t(img)

        return {'img': img, 'phrases': phrases}
    

class MyCollate: 
    
    def __init__(self, 
    tokenizer): 

        super().__init__()
        self.tokenizer = tokenizer
        self.pad_idx = self.tokenizer.pad_token_id

    def __call__(self, batch): 
 
        imgs = [item['img'] for item in batch] 
        imgs = torch.cat(imgs, dim=0)

        phrases = [item['phrases'] for item in batch]
        phrases = torch.cat(phrases, dim=0)

        return imgs, phrases
         
def build(root, 
         sett, 
         resolution, 
         transform: list): 
    
    dataset = VGDataset(root=root, resolution=resolution, transform=transform) 
    return dataset 
