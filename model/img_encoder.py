# image encoder, outputs image slots
import torch 
import torch.nn as nn 
import numpy as np

from .slot_attention import SlotAttention

# soft positional embeddings with learnable projection
#taken from official implementation
class SoftPositionEmbed(nn.Module): 

    def __init__(self, 
            hidden_size: int, # size of input feat dimension
            resolution: tuple): # tuple of ints, H, C of grid 

        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.grid = build_grid(resolution) 
    
    def call(self, 
            inputs): 
        return inputs+self.dense(self.grid) 

# making a grid for input in learnable pos embed
def build_grid(resolution: tuple): 
    '''
    resolution = tuple of H, B  
    '''

    ranges = [np.linspace(0., 1., num=res) for res in resolution] 
    #ranges[0].shape = (res[0],) 
    grid = np.meshgrid(*ranges, sparse=False, indexing='ij')
    # change the ranges into list grid[0]-x axis, grid[1]-y axis
    grid = np.stack(grid, axis=-1) 
    #stack the grids along last axis, shape = H,W,2
    grid = np.expand_dims(grid, axis=0) 
    #equivalent to grid.unsqueeze(0)
    grid = grid.astype(np.float32)
    ##########################################
    return np.concatenate([grid, 1.0-grid], axis=-1) 
    ##########################################

class SlotImage(nn.Module): 

    def __init__(self, 
            resolution: tuple, # tuple H, W
            num_slots: int, #no. of slots (k) 
            num_iter: int, 
            slot_dim: int): #no. of iterations (t)

        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iter = num_iter
        self.slot_dim = slot_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 5,  padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding='same'),  
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding='same'), 
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding='same'), 
            nn.ReLU()
            )        # size = N, 64, H, W, 

        self.pos_emb = SoftPositionEmbed(64, self.resolution) 

        self.mlp = nn.Sequential(
            nn.Linear(64, self.slot_dim), 
            nn.ReLU(), 
            nn.Linear(self.slot_dim, self.slot_dim)
            )

        #####################################################
        self.layer_norm = nn.LayerNorm(self.slot_dim)#layernorm along the embedding dim 
        #####################################################

        self.slot_attention_module = SlotAttention(num_slots=self.num_slots, 
                                                iters=self.num_iter,
                                                dim=self.slot_dim, 
                                                hidden_dim=self.resolution[0]*self.resolution[1])

    def forward(self, 
            inp: torch.tensor): # inp image after transform
        '''
        inp.shape = N, C, H, W 
        '''
        x = inp
        res = (x.shape[2], x.shape[3])
        #print(inp.shape) 
        x = self.encoder(inp) 
        x = x.reshape(x.shape[0], x.shape[1], -1) #x.shape = N,C, H*W 
        x = x.permute(0, 2, 1) #x.shape = N, H*W, C 
        x = self.layer_norm(self.mlp(x)) 
        x = self.slot_attention_module(x) 
        
        return x         
