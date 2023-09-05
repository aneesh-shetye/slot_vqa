'''
be careful while defining the transforms
'''
import pandas as pd 
import cv2

import numpy as np
from skimage import draw 

import torch
from torchvision import transforms as T

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import CLIPTokenizer

class PhrasecutDataset(Dataset): 

    def __init__(self, 
            root, 
            sett, 
            resolution,
            transform):  

        super().__init__()        
        self.root = root
        self.df = pd.read_json(f'{self.root}/refer_{sett}.json')
        self.T = transform
        self.res = resolution
        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')

    
    def __len__(self): 
        return len(self.df['image_id'])

    def __getitem__(self, index: int): 

        img_id = self.df['image_id'][index]
        img = cv2.imread(f'{self.root}/images/{img_id}.jpg')
        img = torch.tensor(img)
        img = img.permute(-1, 0, 1)
        # print(f'img.shape: {img.shape}')
        phrase = self.df['phrase'][index] 
        # print(phrase)
        pol = np.array(self.df['Polygons'][index][0][0]) 
        # print(f'pol.shape: {pol.shape}')
        mask = draw.polygon2mask((img.shape[2], img.shape[1]),pol)
        # print(f'mask.type = {type(mask)}')
        mask = torch.tensor(mask)
        mask = torch.zeros(mask.shape).masked_fill(mask==True, 1.)
        ############################################
        mask = mask.permute(-1,0)
        ############################################
        # print('mask converted to tensor')
        # print(f'mask.shape before repeat{mask.shape}')
        mask = mask.unsqueeze(0).repeat(3,1,1)
        # print(f'mask.shape after repeat{mask.shape}')

        resize = T.Resize(self.res)
        # print(f'image.shape before resize:{img.shape}')
        img = resize(img)        
        # print(f'image.shape after resize:{img.shape}')
        mask = resize(mask)

        for t in self.T: 
            # print(img.shape)
            img = t(img)
                    

        return {'img': img, 'phrase': phrase, 'mask': mask}

class MyCollate: 
    
    def __init__(self, 
    tokenizer): 

        super().__init__()
        self.tokenizer = tokenizer
        self.pad_idx = self.tokenizer.pad_token_id

    def __call__(self, batch): 
 
        imgs = [item['img'] for item in batch] 
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
         sett, 
         resolution, 
         transform: list): 
    
    dataset = PhrasecutDataset(root=root, sett=sett, resolution=resolution, transform=transform) 
    return dataset 



