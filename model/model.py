import torch 
import torch.nn as nn
from transformers import AutoTokenizer, BertModel, CLIPVisionModel, CLIPTextModel 

from .img_encoder import SlotImage
from .text_encoder import SlotText 
from .decoder import SlotDecoder

import matplotlib.pyplot as plt 

"""
NOTE: Be careful about the resolution that is being passed
resolution should be equal to transform applied 
"""

class SlotVQA(nn.Module): 

    def __init__(self, 
            clip_vision_model, 
            mbert, 
            mbert_out_size: int =512, 
            img_enc_out_size: int=768, 
            resolution: tuple =(224, 224), 
            slots_img: int =5, 
            iters_img: int =5, 
            slot_dim_img: int =64, 
            slots_text: int =5, 
            iters_text: int =5, 
            slot_dim_text: int =64, 
            num_head: int =6, 
            transf_dim: int =256, 
            transf_num_layers: int =3, 
            ans_dim: int =200): 

        super().__init__()
        self.clip_vision_model = clip_vision_model
        self.img_enc_out_size = img_enc_out_size
 
        self.mbert = mbert 
        self.mbert_out_size = mbert_out_size

        self.res = resolution
        self.slots_img = slots_img
        self.iters_img = iters_img
        self.slot_dim_img = slot_dim_img
        self.slots_text = slots_text
        self.iters_text = iters_text
        self.slot_dim_text = slot_dim_text
        self.transf_dim = transf_dim

        self.img_enc = SlotImage(self.clip_vision_model, 
                resolution=self.res, 
                mbert_out_size=self.img_enc_out_size,num_slots=self.slots_img, num_iter=self.iters_img, 
                slot_dim=self.slot_dim_img)
            
        self.text_enc = SlotText(mbert=self.mbert, 
                    num_slots=self.slots_text, num_iter=self.iters_text, 
                    mbert_out_size=self.mbert_out_size, slot_dim=self.slot_dim_text) 

        transf_layer = nn.TransformerEncoderLayer(d_model=transf_dim, nhead=num_head, batch_first=True)
        self.transf_enc = nn.TransformerEncoder(transf_layer, num_layers=transf_num_layers)

        # self.rnn = nn.RNN(input_size=transf_dim, hidden_size=transf_dim, batch_first=True, num_layers=2)

        #project text_slots to common embedding space
        self.linear_text = nn.Linear(self.slot_dim_text, transf_dim)
        self.to_img_dim = nn.Linear(self.mbert_out_size, self.slot_dim_img)
        #project img_slots to common embedding space
        self.linear_img = nn.Linear(self.slot_dim_img, transf_dim)

        self.linear_to_transf = nn.Linear(transf_dim*2, transf_dim)

        self.linear_emb_to_img = nn.Linear(self.mbert_out_size , self.slot_dim_img)

        #linear for cls type token (just like a look-up table) 
        self.learnable_cls = nn.Linear(self.transf_dim, transf_dim)
        # self.layernorm_slots = nn.LayerNorm(transf_dim)

        #output of cls to ans
        # self.ans_enc = nn.Sequential(
        #                     nn.Linear(self.transf_dim, self.transf_dim//2), 
        #                     nn.ReLU(), 
        #                     nn.Linear(self.transf_dim//2, self.transf_dim//4), 
        #                     nn.ReLU(), 
        #                     nn.Linear(self.transf_dim//4, self.transf_dim//8), 
        #                     nn.ReLU(), 
        #                     nn.Linear(self.transf_dim//8, ans_dim)) 
                            #nn.Softmax(dim=-1)) #crossentropyloss would take care of this 
        self.decoder = SlotDecoder(decoder_init_size=self.res, 
                                    slot_dim=transf_dim)
        self.ans_enc = nn.Sequential(
                            nn.Linear(self.transf_dim, ans_dim)) 
        self.layernorm = nn.LayerNorm(ans_dim)

    def forward(self, 
            img: torch.tensor, 
            text: torch.tensor, 
            return_text_att: bool=False, 
            object_seg: bool=False): 

        
        self.device = next(self.learnable_cls.parameters()).device
        text_emb, cls, text_att, text_slots = self.text_enc(text) 
        # text_slots = self.to_img_dim(text_slots)
        # guide = text_slots
        guide = self.to_img_dim(cls.unsqueeze(1).repeat(1,self.slots_img,1))
        # print(f'guide.shape: {guide.shape}')
        # print(f'cls.shape: {cls.shape}')
        # print(torch.sum(text_emb, dim=1).shape)
        # guide = self.to_img_dim(torch.sum(text_emb, dim=1)/text_emb.shape[1])
        # print(f'guide.shape:{guide.shape}')
        # print(text_emb.shape)
        # guide = self.to_img_dim(torch.sum(text_emb, dim=1))
        img_emb, img_slots = self.img_enc(inp=img, guide=guide) 
        #img_slots.shape = batch, num_slots, slot_dim 

        #projecting imgs and texts to a common embedding space
        text_slots = text_emb
        text_slots = self.linear_text(text_slots)
        img_slots = self.linear_img(img_slots)

        ################################################
        #IMG RECONSTRUCTION: 
        ################################################
        
        if object_seg:  
            recon_combined, masks, recons = self.decoder(img_slots, self.device)
            return recon_combined, masks, recons
        ################################################
        #AGREGATING IMAGE SLOTS
        ################################################

        # k = img_slots.shape[1]
        # img_slots = (torch.sum(img_slots, dim=1)/k).unsqueeze(1)
        ################################################

        ###################################################
        #CONCAT AT DIM=-1
        ###################################################
        # comb_slots = torch.cat((text_slots, img_slots.repeat(1,text_slots.shape[1], 1)), dim=-1)
        # comb_slots = self.linear_to_transf(comb_slots)
        ###################################################

        ###################################################
        # CONCAT AT DIM=1
        ###################################################
        comb_slots = torch.cat((text_slots, img_slots), dim=1)
        # comb_slots.shape = batch,num_slots_text + num_slots_img, slot_dim 
        ####################################################
        
        ###################################################
        #PREPEND WITH CLS TOKEN 
        ###################################################
        
        # cls_tok = torch.ones((img_slots.shape[0], self.transf_dim), dtype=torch.float).to(self.device) 
        # #cls_tok.shape = batch_size, transf_dim
        # cls_tok = self.learnable_cls(cls_tok)        
        # # print(cls_tok.shape) 
        # comb_slots = torch.cat((cls_tok.unsqueeze(1), comb_slots), dim=1) 
        #comb_slots.shape = batch, num_slots_text + num_slots_img + 1, slot_dim
        ####################################################

        out = self.transf_enc(comb_slots)
        # out = comb_slots
        
        ###################################################
        #RNN ENCODER: 
        ###################################################
        # _, out = self.rnn(comb_slots)
        # out = out.permute(1, 0, 2)
        # ans = out.reshape(out.shape[0], -1)
        ###################################################

        ##################################################
        #AGREGATING STRATEGY FOR FINAL ANS: SUM  
        ##################################################
        # ans = torch.sum(out, dim=1)/out.shape[1]
        #################################################

        ##################################################
        #AGREGATING STRATEGY FOR FINAL ANS: CLS  
        ##################################################
        ans = out[:, 0, :]
        #################################################
        # return self.layernorm(self.ans_enc(ans))
        if return_text_att: 
            return text_att, self.ans_enc(ans)
        else: 
            return self.ans_enc(ans)
        


 
