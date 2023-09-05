from easy_vqa import get_train_questions, get_test_questions
from easy_vqa import get_train_image_paths, get_test_image_paths

import torch
from torch.utils.data import DataLoader, Dataset 
from torch.nn.utils.rnn import  pad_sequence

import PIL 
from PIL import Image
import cv2

def build_ans2id(answers_list): 

    ans2id = set(answers_list)
    ans2id = {i:n for n,i in enumerate(ans2id)}
    return ans2id

class EasyVQA(Dataset):

  def __init__(self,
               get_train_questions:list,
               get_train_answers: list, 
               get_train_image_ids: list,
               get_train_image_paths,
               tokenizer,  
               transform): 
    super().__init__() 
    self.get_train_questions = get_train_questions
    self.get_train_answers = get_train_answers
    self.get_train_image_ids = get_train_image_ids
    self.get_train_image_paths = get_train_image_paths
    self.tokenizer = tokenizer
    self.T = transform 
    # print(self.get_train_answers)
    self.ans2id = build_ans2id(self.get_train_answers)

  def __len__(self):
    return len(self.get_train_image_ids) 

  def __getitem__(self, index: int): 
    
    img_id = self.get_train_image_ids[index]
    img_path = self.get_train_image_paths[img_id]
    img = Image.open(img_path)
    # img = cv2.imread(img_path)
    for t in self.T: 
      img = t(img)

    #normalizing
    img = (img-img.mean())/img.std()

    ques = self.get_train_questions[index]

    answer = self.ans2id[self.get_train_answers[index]]

    return {"img": img, 
            "ques": ques, 
            "ans": answer}


class MyCollate: 

    def __init__(self, 
                 tokenizer): 

        self.tokenizer = tokenizer
        self.pad_idx = self.tokenizer.pad_token_id
    
    def __call__(self, batch): 

        # imgs = [item['img'] for item in batch] 
        imgs = [item['img'] for item in batch]
        imgs = torch.cat(imgs, dim=0)
        
        ques = []
        for item in batch: 
            ques.append(self.tokenizer(item['ques'].lower(), padding=True,   
                        max_length=512, return_tensors='pt',truncation=True)['input_ids'].T)

        questions = pad_sequence(ques, batch_first=True, 
                padding_value=self.pad_idx)

        ans = torch.tensor([item['ans'] for item in batch])

        return imgs, questions, ans 

def build(train_questions: list, 
        train_answers: list,
        train_images_ids: list,
        train_image_paths: list, 
        tokenizer, 
        transform):  


    dataset = EasyVQA(get_train_questions=train_questions,
               get_train_answers=train_answers, 
               get_train_image_ids=train_images_ids,
               get_train_image_paths= train_image_paths,
               tokenizer=tokenizer,  
               transform = transform)
    return dataset


