# image encoder, outputs image slots
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from transformers import CLIPVisionModel

from .slot_attention import SlotAttention

from .utils import SoftPositionEmbed

class SlotImage(nn.Module): 

    def __init__(self, 
            clip_vision_model, 
            resolution: tuple, # tuple H, W
            mbert_out_size:int, 
            num_slots: int, #no. of slots (k) 
            num_iter: int, 
            slot_dim: int, 
            add_cls: bool =True): #no. of iterations (t)

        super().__init__()
        self.resolution = resolution
        self.mbert_out_size = mbert_out_size
        self.num_slots = num_slots
        self.num_iter = num_iter
        self.slot_dim = slot_dim
        self.add_cls = add_cls

        # self.encoder = nn.Sequential(
        #     nn.Conv2d(3, 64, 5,  padding='same'),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, 5, padding='same'),  
        #     nn.ReLU())
            # nn.Conv2d(64, 64, 5, padding='same'), 
            # nn.ReLU(),
            # nn.Conv2d(64, 64, 5, padding='same'), 
            # nn.ReLU()
            # )        # size = N, 64, H, W, 

        self.clip_encoder = clip_vision_model

        self.pos_emb = SoftPositionEmbed(self.mbert_out_size, self.resolution) 
        # self.pos_emb = SoftPositionEmbed(64, self.resolution) 

        self.mlp = nn.Sequential(
            nn.Linear(self.mbert_out_size, self.slot_dim), 
            nn.ReLU(), 
            nn.Linear(self.slot_dim, self.slot_dim)
            )

        #####################################################
        self.layer_norm = nn.LayerNorm(self.slot_dim)#layernorm along the embedding dim 
        #####################################################

        self.slot_attention_module = SlotAttention(num_slots=self.num_slots, 
                                                iters=self.num_iter,
                                                dim=self.slot_dim, 
                                                hidden_dim=self.slot_dim)

    def forward(self, 
            inp: torch.tensor, 
            slots=None,
            guide=None, 
            add_cls: bool =True): # inp image after transform
        '''
        inp.shape = N, C, H, W 
        '''
        x = inp
        # print(f'image input.shape = {inp.shape}')
        # res = (x.shape[2], x.shape[3])
        #print(inp.shape) 
        # print(f'image input shape = {x.shape}')
        # x = self.encoder(inp) 
        # print(x.shape)
        # self.add_cls = add_cls
        x = self.clip_encoder(x)#x.shape = batch_size, 50, 768
        # print(f'shape of image input after encoder = {x.shape}')

        '''
        try appending cls tok to each embedding
        '''

        add_cls = False
        
        if add_cls: 
            cls_token = x['pooler_output'] #pooler_output.shape = batch_size, 768

        # print(x['last_hidden_state'].shape)
        img_emb = x['last_hidden_state'] #img_emb.shape = batch_size, 50, img_dim          
        # print(f'img_emb.shape = {img_emb.shape}')

        if add_cls:             
            cls_token = cls_token.unsqueeze(1).repeat(1,img_emb.shape[1], 1)
            img_emb = torch.cat((cls_token, img_emb), dim=-1) 

        # x = self.pos_emb(x)
        # x = x.reshape(x.shape[0], x.shape[1], -1) #x.shape = N,C, H*W 
        # x = x.permute(0, 2, 1) #x.shape = N, H*W, C 

        # print(f'x shape in img encoder before mlp{x.shape}')
        img_slots = self.layer_norm(self.mlp(img_emb)) 
        if guide==None: 
            img_att , img_slots = self.slot_attention_module(inputs=img_slots, slots=slots) 
        else: 
            img_att , img_slots = self.slot_attention_module(inputs=img_slots, slots=slots, guide=guide)
        
        return img_att, img_emb, img_slots
