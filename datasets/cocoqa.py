from PIL import Image

from torch import tensor
from torch.utils.data import Dataset

from transformers import CLIPTokenizer

class CocoQADataset(Dataset): 

    def __init__(self,
            root, 
            imset, 
            tokenizer, 
            transform): 

        self.root = root
        self.imset = imset
        self.tokenizer = tokenizer
        self.transform = transform

        self.img_file = open(f'{root}{imset}/img_ids.txt')
        self.ques_file = open(f'{root}{imset}/questions.txt')
        self.ans_file = open(f'{root}{imset}/answers.txt')

        self.img_ids = self.img_file.readlines() 
        self.questions = self.ques_file.readlines() 
        self.answers = self.ans_file.readlines() 

        self.ans_set = set(self.answers)
        self.answer2id = {i[:-1]: n for n, i in enumerate(self.ans_set)}

    def __len__(self): 
        return len(self.answers)

    def __getitem__(self, 
    index) -> dict:

        img_id = self.img_ids[index][:-1]
        try:         
            img = Image.open(f'{self.root}{self.imset}2017/000000{img_id}.jpg')
        except:
            print('image missing')
        for t in self.transform: 
            img = t(img)
        ques = self.questions[index][:-1] 
        ans = self.answer2id[self.answers[index][:-1]]

        return {img: img, 
                ques: ques, 
                ans: ans} 
    

def build( transform, 
        args):  

    tokenizer = CLIPTokenizer.from_pretrained(args.text_encoder_type)

    dataset = CocoQADataset(root=args.root,
               imset=args.imset,
               tokenizer=tokenizer,  
               transform = transform)
    return dataset

         





