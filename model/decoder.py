#region: LIBRARY IMPORTS
from termios import INPCK
import torch 
import torch.nn as nn 

from .utils import SoftPositionEmbed 

#endregion

#region: HELPER FUNCTIONS:  
def spatial_broadcast(slots, res): 
    """   
    Broadcast slot features to a 2D grid and collapse slot dimension
    """
    slots = slots.reshape(-1, slots.shape[-1])
    grid = slots.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, res[0], res[1])

    return grid

def unstack_and_split(x, batch_size: int, num_channels=3): 
    """
    similar to the original slot_attention paper code
    Unstack batch dimension and split into channels and alpha mask
    """

    unstacked = x.reshape([batch_size, -1, x.shape[1], x.shape[2], x.shape[3]])
    # print(f'unstacked.shape: {unstacked.shape}')
    channels, mask = torch.split(unstacked, [num_channels, 1], dim=2)
    # print(f'channels.shape = {channels.shape}')
    # print(f'mask.shape = {mask.shape}')
    return channels, mask

#endregion
    
class SlotDecoder(nn.Module): 
    
    def __init__(self, 
        decoder_init_size, #list/tuple
        slot_dim: int, 
    ): 
        super().__init__()
        self.decoder_init_size = decoder_init_size
        self.decoder_pos = SoftPositionEmbed(slot_dim, self.decoder_init_size)

        #transpose Conv Decoder: 
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(slot_dim, 256, 128, padding=(2,2)),  # 256
            nn.ReLU(), 
            nn.ConvTranspose2d(256, 64, 5, padding=(2,2)), #64
            nn.ReLU(), 
            nn.ConvTranspose2d(64, 16, 5, padding=(2,2)), #16
            nn.ReLU(), 
            nn.ConvTranspose2d(16, 4, 5, padding=(2,2)), #4
            nn.ReLU(), 
            nn.ConvTranspose2d(256, 4, 5, padding=(2,2)), #4
            nn.ReLU() 
        )

    
    def forward(self, 
                inp: torch.tensor, 
                device): 
        
        batch_size = inp.shape[0]

        inp = spatial_broadcast(inp, self.decoder_init_size)
        #inp.shape = B*num_slots, H, W, D_slots
        inp = self.decoder_pos(inp, device)

        # print(f'inp.shape after spatial broadcast {inp.shape}')
        x = self.decoder_cnn(inp)
        #x.shape = B*N_s, 4, H, W

        # print(f'cnn output.shape: {x.shape}')
        recons, masks = unstack_and_split(x, batch_size = batch_size)
        #recons.shape = B, N_s, 3, H, W
        #masks.shape = B, N_s, 1, H, W 

        #normalizing alpha masks over slots
        masks = torch.softmax(masks, axis=1)        
        recon_combined = torch.sum(recons*masks, axis=1) #axis=N_s
        #recon_combined.shape = B, 3, H, W

        return recon_combined, recons, masks
        
