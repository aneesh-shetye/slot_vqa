
import pandas as pd 
import cv2

import numpy as np
from skimage import draw 

import torch

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
class PhrasecutDataset(Dataset): 

    def __init__(self, 
            root, 
            transform):  

        super().__init__()        
        self.root = root
        self.df = pd.read_json(f'{self.root}/refer_miniv.json')
        self.T = transform

    
    def __len__(self): 
        return len(self.df['image_id'])

    def __getitem__(self, index: int): 

        img_id = self.df['image_id'][index]
        img = cv2.imread(f'/home/aneesh/datasets/PhraseCutDataset/data/VGPhraseCut_v0/images/{img_id}.jpg')
        img = torch.tensor(img)
        img = img.permute(-1, 0, 1)
        # print(f'img.shape: {img.shape}')
        phrase = self.df['phrase'][index] 
        # print(phrase)
        pol = np.array(self.df['Polygons'][index][0][0]) 
        # print(f'pol.shape: {pol.shape}')
        mask = draw.polygon2mask((img.shape[0], img.shape[1]),pol)
        # print(f'mask.type = {type(mask)}')
        mask = torch.tensor(mask)
        mask = torch.ones(mask.shape).masked_fill(mask==True, 0.)
        # print('mask converted to tensor')
        mask = mask.unsqueeze(-1).repeat(1,1,3)
        for t in self.T: 
            img = t(img)
            mask = t(mask)

        return {'img': img, 'phrase': phrase, 'mask': mask}

class MyCollate: 
    
    def __init__(self, 
    tokenizer): 

        super().__init__()
        self.tokenizer = tokenizer
        self.pad_idx = self.tokenizer.pad_token_id

    def __call__(self, batch): 
 
        imgs = [item['img'].unsqueeze(0) for item in batch] 
        imgs = torch.cat(imgs, dim=0)

        phrases = []
        for item in batch:
            phrases.append(self.tokenizer(item['phrase'].lower(), padding=True, 
                    max_length=512, return_tensors='pt', truncation=True)['input_ids'].T)
        phrases = pad_sequence(phrases, batch_first=True)

        masks = [item['mask'].unsqueeze(0) for item in batch] 
        masks = torch.cat(masks, dim=0)

        return imgs, phrases, masks
         


def build(root, 
         transform: list): 
    
    dataset = PhrasecutDataset(root=root, transform=transform) 

    return dataset 
