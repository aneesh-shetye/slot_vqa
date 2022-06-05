import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset 
import torchvision.transforms as transforms 
from datasets import load_dataset
from transformers import AutoTokenizer

class VG_dataset(Dataset): 

    def __init__(self, 
                 transformations: list = None): 

        self.transformations = transformations 
        self.dataset = load_dataset("visual_genome", "question_answers_v1.2.0")
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')

        self.img_list = []
        self.ques_list = []
        self.ans_list = []

        self.mean, self.std = self.calc_stats(self.dataset)
        if self.transformations is not None: 
            self.transformations.append(transforms.Normalize(self.mean, self.std)) 

        else: 
            self.transformations = [transforms.Normalize(self.mean, self.std)]
        T = transforms.Compose(self.transformations)

        for n, i in enumerate(self.dataset['train']): 
            
            img = np.array(i['image'])

            for t in i['qas']:                 
                self.img_list.append(T(img))
                self.ques_list.append(self.tokenizer(t['question'].lower(), padding=True, max_length=512, return_tensors='pt', truncation=True)['input_ids'])
                self.ans_list.append(t['answer'])
            
            if n==100: 
                break
        print("======out of for loop========")
        self.ans_dict = self.build_ans_dict(self.ans_list)
    

    def __len__(self): 
        return 101 #len(self.ans_list)

    def __getitem__(self, index: int): 
        img = self.img_list[index]
        ques = self.ques_list[index]
        ans = torch.tensor(self.ans_dict[self.ans_list[index]])
        
        return {'img': img, 
                'ques': ques, 
                'ans': ans}
    
    def build_ans_dict(self, 
                       ans: list): 
        
        ans_set = set(ans)
        self.ans_dict = { a.lower():i for i,a in enumerate(ans_set)}
        return self.ans_dict 

    def calc_stats(self, 
            dataset): 

        # finding mean and std of images: 
        height = []
        width = []

        c0 = []
        c1 = []
        c2 = []

        for n,i in enumerate(dataset['train']): 
                # print(i['qas'][0]['question'])
                # print(i['qas'][0]['answer'])
            img = np.array(i['image'])
            # images.append(i['image'])
            img = torch.tensor(img)

            tot = torch.sum(img[:,:, 0])/(img.shape[0]*img.shape[1])
            c0.append(tot.item())
            tot = torch.sum(img[:,:, 1])/(img.shape[0]*img.shape[1])
            c1.append(tot.item())
            tot = torch.sum(img[:,:, 2])/(img.shape[0]*img.shape[1])
            c2.append(tot.item())
            # print(img.shape)

            height.append(img.shape[0])
            width.append(img.shape[1])
# for first 100 images for now 
            if n == 100: 
                break 

        c0 = np.array(c0)
        c1 = np.array(c1)
        c2 = np.array(c2)

        mean = (c0.mean(), c1.mean(), c2.mean())
        std = (c0.std(), c1.std(), c2.std())
        
        return mean, std

class MyCollate: 

    def __init__(self, 
                 tokenizer): 

        self.tokenizer = tokenizer
        self.pad_idx = self.tokenizer.pad_token_id
    
    def __call__(self, batch): 

        imgs = [item['img'].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0 )
        
        ques = []
        for item in batch: 
            ques.append(item['ques'].T)
        questions = pad_sequence(ques, batch_first=True, padding_value=self.pad_idx)

        ans = [item['ans'] for item in batch]

        return imgs, questions, ans 

'''
example transform: 
resize all images to 600. 600 and normalize the pixels: 

t = transforms.compose([transforms.functional.to_tensor,
                        transforms.resize([600, 600]), 
                    transforms.normalize(mean, std)]) 
'''

