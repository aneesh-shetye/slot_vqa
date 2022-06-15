import torch 
import torch.nn as nn
from transformers import AutoTokenizer, BertModel 

from .img_encoder import SlotImage
from .text_encoder import SlotText 

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

        self.img_enc = SlotImage(self.clip_vision_model, resolution=self.res, 
                mbert_out_size=self.img_enc_out_size,num_slots=self.slots_img, num_iter=self.iters_img, 
                slot_dim=self.slot_dim_img)
            
        self.text_enc = SlotText(mbert=self.mbert, 
                    num_slots=self.slots_text, num_iter=self.iters_text, 
                    mbert_out_size=self.mbert_out_size, slot_dim=self.slot_dim_text) 

        transf_layer = nn.TransformerEncoderLayer(d_model=transf_dim, nhead=num_head, batch_first=True)
        self.transf_enc = nn.TransformerEncoder(transf_layer, num_layers=transf_num_layers)

        #project text_slots to common embedding space
        self.linear_text = nn.Linear(self.slot_dim_text, transf_dim)
        #project img_slots to common embedding space
        self.linear_img = nn.Linear(self.slot_dim_img, transf_dim)

        #linear for cls type token (just like a look-up table) 
        self.learnable_cls = nn.Linear(self.transf_dim, transf_dim)

        #output of cls to ans
        self.ans_enc = nn.Sequential(nn.Linear(self.transf_dim, ans_dim), 
                            nn.ReLU())#, 
                            # nn.Linear(ans_dim, ans_dim)) 
                            #nn.Softmax(dim=-1)) #crossentropyloss would take care of this 
        self.layernorm = nn.LayerNorm(ans_dim)

    def forward(self, 
            img: torch.tensor, 
            text: torch.tensor): 

        self.device = next(self.learnable_cls.parameters()).device
        # print('inside model:')
        # print(f'img.shape: {img.shape}')
        # print(f'text.shape: {text.shape}')
        text_slots = self.text_enc(text) 
        img_slots = self.img_enc(img) 
        #slots.shape = batch, num_slots, slot_dim 

        #projecting imgs and texts to a common embedding space
        text_slots = self.linear_text(text_slots)
        img_slots = self.linear_img(img_slots)

        # print(f'text slots.shape = {text_slots.shape}, img_slots.shape = {img_slots.shape}')
        comb_slots = torch.cat((text_slots, img_slots), dim=1)
        # print(f'comb_slots.shape: {comb_slots.shape}')
        #comb_slots.shape = batch,num_slots_text + num_slots_img, slot_dim 
        #preprend this with a cls 
        
        '''
        cls_tok = torch.ones((img_slots.shape[0], self.transf_dim), dtype=torch.float).to(self.device) 
        #cls_tok.shape = batch_size, transf_dim
        cls_tok = self.learnable_cls(cls_tok)        
        # print(cls_tok.shape) 
        comb_slots = torch.cat((cls_tok.unsqueeze(1), comb_slots), dim=1) 
        # print(f'comb_slots+cls.shape: {comb_slots.shape}')
        #comb_slots.shape = batch, num_slots_text + num_slots_img + 1, slot_dim
        '''

        out = self.transf_enc(comb_slots)
        # print(f'out.shape: {out.shape}')
        # print(f'out=======================================> {out}')
        ans = out[:,0, :]
        # print(f'ans=====================================> {ans}')

        return self.layernorm(self.ans_enc(ans))


 

