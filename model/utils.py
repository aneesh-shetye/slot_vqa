import torch
from torch import nn
import numpy as np

# soft positional embeddings with learnable projection
#taken from official implementation
class SoftPositionEmbed(nn.Module): 

    def __init__(self, 
            hidden_size: int, # size of input feat dimension
            resolution: tuple): # tuple of ints, H, C of grid 

        super().__init__()
        self.dense = nn.Linear(4, hidden_size)
        self.resolution = resolution
    
    def forward(self, 
            inputs, 
            device): 
        self.grid = self.build_grid(device)
        # print(f'self.grid.shape: {self.grid.shape}')
        # print(f'inputs.shape: {inputs.shape}')
        return inputs+self.dense(self.grid).permute(-1,0,1).unsqueeze(0) 

    # making a grid for input in learnable pos embed
    def build_grid(self, 
            device): 
        '''
        resolution = tuple of H, B  
        '''

        resolution = self.resolution
        ranges = [np.linspace(0., 1., num=res) for res in resolution] 
        #ranges[0].shape = (res[0],) 
        grid = np.meshgrid(*ranges, sparse=False, indexing='ij')
        # change the ranges into list grid[0]-x axis, grid[1]-y axis
        grid = np.stack(grid, axis=-1) 
        #stack the grids along last axis, shape = H,W,2
        # grid = np.expand_dims(grid, axis=0) 
        #equivalent to grid.unsqueeze(0)
        grid = grid.astype(np.float32)
        ##########################################
        return torch.tensor(np.concatenate([grid, 1.0-grid], axis=-1)).to(device) 
        ##########################################