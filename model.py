import torch 
import torch.nn as nn
from transformers import AutoTokenizer, BertModel 

from img_encoder import SlotImage
from text_encoder import SlotText 

class SlotVQA(nn.Module): 

    def __init__(self, 
            mbert, 
            mbert_out_size: int =768, 
            resolution: tuple =(600, 600), 
            num_slots: int =5, 
            num_iter: int =5, 
            slot_dim: int =64, 
            num_head: int =6, 
            transf_dim: int =256, 
            transf_num_layers: int =3): 

        self.mbert = mbert, 
        self.mbert_out_size = mbert_out_size

        self.res = resolution
        self.slots = num_slots
        self.iters = num_iter
        self.slot_dim = slot_dim 

        self.img_enc = SlotImage(resolution=self.res, 
                num_slots=self.slots, num_iter=self.iters, 
                slot_dim=self.slot_dim)
            
        self.text_enc = SlotText(mbert=self.mbert, 
                    num_slots=self.slots, num_iter=self.iters, 
                    mbert_out_size=self.mbert_out_size, slot_dim=self.slot_dim) 

        transf_layer = nn.TransformerEncoderLayer(d_model=transf_dim, nhead=nhead)
        transf_enc = nn.TransformerEncoder(encoder_layer, num_layers=transf_num_layers)


    def forward(self, 
            img: torch.tensor, 
            text: torch.tensor): 

        text_slots = self.text_enc(text) 
        img_slots = self.img_enc(img) 




 