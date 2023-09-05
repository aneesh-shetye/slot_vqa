'''
generate captions from slots
NOTE: Mind the tokenizer. It should be same as the one used in Dataset
'''

import torch 
import torch.nn as nn
from transformers import AutoTokenizer, BertModel, CLIPVisionModel, CLIPTextModel 

from .img_encoder import SlotImage
from .text_encoder import SlotText 

from transformers import GPT2LMHeadModel,  GPT2Tokenizer



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
            ans_dim: int =200, 
            decoder_dim: int=768, 
            out_vocab: int=2000): 

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
        self.decoder_dim = decoder_dim 

        self.img_enc = SlotImage(self.clip_vision_model, 
                resolution=self.res, 
                mbert_out_size=self.img_enc_out_size,num_slots=self.slots_img, num_iter=self.iters_img, 
                slot_dim=self.slot_dim_img)
            
        self.text_enc = SlotText(mbert=self.mbert, 
                    num_slots=self.slots_text, num_iter=self.iters_text, 
                    mbert_out_size=self.mbert_out_size, slot_dim=self.slot_dim_text) 

        transf_layer = nn.TransformerEncoderLayer(d_model=transf_dim, nhead=num_head, batch_first=True)
        self.transf_enc = nn.TransformerEncoder(transf_layer, num_layers=transf_num_layers)

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

        self.ans_enc = nn.Sequential(
                            nn.Linear(self.transf_dim, ans_dim)) 
        self.layernorm = nn.LayerNorm(ans_dim)

        self.linear_to_decoder = nn.Linear(slot_dim_img, self.decoder_dim)

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gpt_decoder = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

        self.to_vocab = nn.Linear(self.decoder_dim, out_vocab) 

    def forward(self, 
            img: torch.tensor, 
            seq_len: int, 
            att_mask: torch.tensor=None): 
        
        #trg.shape = batch_size, seq_len,  
        
        self.device = next(self.learnable_cls.parameters()).device
        #text_emb.shape = batch_size, seq_len, emb_dim
        # text_slots = self.to_img_dim(text_slots)
        # guide = text_slots
        img_att, img_emb, img_slots = self.img_enc(inp=img, guide=guide) 
        #img_slots.shape = batch, num_slots, slot_dim 

        #projecting image slots to decoder space
        img_slots = self.linear_to_decoder(img_slots)

        #projecting imgs and texts to a common embedding space
        # print(f'text_emb.shape:{text_emb.shape}')

        num_slots = img_slots.shape[1]
        batch_size = img_slots.shape[0]

        out_slots = torch.zeros((batch_size, num_slots, self.out_vocab))

        for i in range(num_slots): 

            context = img_slots[:, i, :]
            input_embeds = context.unsqueeze(1).repeat(1,seq_len, 1)
            if att_mask == None:
                att_mask = torch.zeros((1, seq_len), dtype=torch.long)
                att_mask[batch_size, 0] = 1

            out = self.gpt_decoder (inputs_embeds = input_embeds, attention_mask = att_mask)           
            out = self.to_vocab(out)
            out_slots[:, i, :] = out
            
        return out_slots

