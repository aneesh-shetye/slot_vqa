#slot attention module from lucidrains
import math
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

#write a class for multihead attention
class MultiheadAttention(nn.Module):
    def __init__(self, 
            query_dim:int,
            key_dim:int, 
            value_dim:int, 
            hidden_dim:int , 
            n_heads:int, 
            dropout = None): 
        super().__init__()
        self.hidden_dim = hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.head_dim = hidden_dim // n_heads
        self.fc_Q = nn.Linear(query_dim, hidden_dim)
        self.fc_K = nn.Linear(key_dim, hidden_dim)
        self.fc_V = nn.Linear(value_dim, hidden_dim)
        self.fc_O = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, query, key, value, mask=None): 

        batch_size = query.shape[0]
        Q = self.fc_Q(query)
        K = self.fc_K(key)
        V = self.fc_V(value)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)     # [Q] = [batch_size, num_heads, query_len, head_dim]
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)     # [K] = [batch_size, num_heads, key_len, head_dim]
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)     # [V] = [batch_size, num_heads, value_len, head_dim]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:                                                            # [energy] = [batch_size, num_heads, query_len, key_len]  
            energy = energy.masked_fill(mask == False, -1e10)                               
        attention = torch.softmax(energy, dim = -1)                                     # [attention] = [batch_size, num_heads, query_len, key_len]
        if self.dropout:
            x = torch.matmul(self.dropout(attention), V)                                    # [x] = [batch_size, num_heads, query_len, head_dim]
        else: 
            x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()                                          # [x] = [batch_size, query_len, num_heads, head_dim]
        # Can avoid contiguous() if we use .reshape instead of .view in the next line
        out = self.fc_O(x.view(batch_size, -1, self.hidden_dim))                        # [out] = [batch_size, query_len, hidden_dim]   
        return out, attention
 
#custom GRU module : 
class customGRU(nn.Module): 

    def __init__(self, 
                inp_dim: int, 
                hidden_dim: int, 
                guide_dim: int): 

        super().__init__()
        self.inp_dim = inp_dim
        self.hidden_dim = hidden_dim
        self.guide_dim = guide_dim

        self.linear_ir = nn.Linear(self.inp_dim, self.hidden_dim)
        self.linear_hr = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.linear_gt = nn.Linear(self.guide_dim, self.hidden_dim)
        self.linear_it = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.linear_iz = nn.Linear(self.inp_dim, self.hidden_dim)
        self.linear_hz = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.linear_in = nn.Linear(self.inp_dim, self.hidden_dim)
        self.linear_hn = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_gn = nn.Linear(self.guide_dim, self.hidden_dim)

    def forward(self, 
                inp, 
                guide, 
                hidden=None): 
        
        if hidden==None: 
            hidden = torch.ones(inp.shape[0], inp.shape[1], self.hidden_dim)

        r = F.sigmoid(self.linear_ir(inp)+self.linear_hr(hidden))
        z = F.sigmoid(self.linear_iz(inp)+self.linear_hz(hidden))
        t = F.sigmoid(self.linear_gt(guide)+self.linear_it(inp))
        n = torch.tanh(self.linear_in(inp)+r*self.linear_hn(hidden)+t*self.linear_gn(guide))
        h = (1-z)*n + z*n

        return h

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)
        self.gru_modified = customGRU(dim, dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        ######################################
        ## change hidden_dim
        ######################################
        self.multihead = MultiheadAttention(dim, dim, dim, hidden_dim=dim, n_heads=8)

    def forward(self, inputs, slots=None, num_slots=None, mask=None, guide=None):
        b, n, d, device = *inputs.shape, inputs.device
        n_s = num_slots if num_slots is not None else self.num_slots
        
        if slots == None: 
            mu = self.slots_mu.expand(b, n_s, -1)
            sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

            slots = mu + sigma * torch.randn(mu.shape, device = device)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            #dots.shape =  batch_size, num_slots, seq_len 
            if mask != None: 
                #mask.shape = batch_size, seq_len
                dots.masked_fill(mask==0, 0.)

            # print(dots.shape) 
            attn = dots.softmax(dim=2) + self.eps
            #attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)


            if guide==None: 
                slots = self.gru(
                    updates.reshape(-1, d),
                    slots_prev.reshape(-1, d)
                )
            
            else: 
                guide, _= self.multihead(slots, guide, guide)
                slots = self.gru_modified(
                    updates.reshape(-1, d), 
                    slots_prev.reshape(-1, d), 
                    guide.reshape(-1, d)
                )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return attn, slots #slots.shape = batch_size, num_slots, dim_slots
