import torch 
import torch.nn as nn
from transformers import AutoTokenizer, BertModel, CLIPTextModel
from .slot_attention import SlotAttention

class SlotText(nn.Module): 

    def __init__(self, 
            mbert, 
            num_slots: int =5, 
            num_iter: int =5, 
            mbert_out_size: int =512, 
            slot_dim: int =64): 

        super().__init__()

        self.mbert_out_size = mbert_out_size
        self.mbert = mbert
        self.num_slots = num_slots
        self.num_iter = num_iter
        self.slot_dim = slot_dim 
        self.layernorm = nn.LayerNorm(self.slot_dim)  

        self.mbert = mbert
        self.mlp = nn.Sequential(
                nn.Linear(self.mbert_out_size, self.slot_dim), 
                nn.ReLU(), 
                nn.Linear(self.slot_dim, self.slot_dim)
                )
        self.slot_attention_module = SlotAttention(num_slots=self.num_slots, 
                                                iters=self.num_iter,
                                                dim=self.slot_dim, 
                                                hidden_dim=self.mbert_out_size)    

    def forward(self, 
             inp: torch.tensor, 
             pad_id: int =0): #B, seq_len 

        device = next(self.slot_attention_module.parameters()).device 
        mask = torch.ones(inp.shape).to(device).masked_fill(inp==pad_id, 0)
        mask = mask.squeeze(-1).unsqueeze(1)
        # print(f'mask.shape={mask.shape}')
        if len(inp.shape)>2: 
            inp = inp.squeeze(-1)
        # print(f'inp.shape={inp.shape}')        
        # print(inp.shape)
        mbert_output = self.mbert(inp)
        x = mbert_output['last_hidden_state']#x.shape = B, seq_len, mbert_out_size
        cls = mbert_output['pooler_output']
        # print(f'x.shape after mbert={x.shape}')
        slots = self.layernorm(self.mlp(x)) #x.shape = B, seq_len, mbert_out_size 
        att, slots = self.slot_attention_module(slots, mask=mask)  
        # slots=x
        
        return x, cls, att, slots 
