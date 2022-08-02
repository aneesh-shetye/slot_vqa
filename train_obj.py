#region: IMPORTING LIBRARIES
import numpy as np
from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time

import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn, optim 
import torch.nn.functional as F 
from torchvision import transforms as T
from transformers import AutoTokenizer, BertModel, CLIPFeatureExtractor, CLIPVisionModel, CLIPTextModel, CLIPTokenizer, CLIPProcessor

from datasets.test_phrasecut import  MyCollate
from datasets.test_phrasecut import build

from model.model import SlotVQA

import wandb 
#endregion

#region: SETTING ENV VARIABLES
os.environ['TRANSFORMERS_OFFLINE'] = 'yes'
os.environ['WANDB_START_METHOD'] = 'thread'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
#endregion

#region: SETTING SEED: 
MANUAL_SEED = 4444

random.seed(MANUAL_SEED)
np.random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)
torch.backends.cudnn.deterministic = True
#endregion

#region: ARGS
parser = argparse.ArgumentParser(description='slot_vqa')

#region: system config: 
parser.add_argument('--workers', default=5, type=int, metavar='N', 
                    help='number of data loader workers') 
parser.add_argument('--print_freq', default=5, type=int, metavar='PF', 
                    help='write in the stats file and print after PF steps') 
parser.add_argument('--checkpoint_dir', default='./checkpoint/', type=Path, metavar='CD', 
                    help='path to directory in which checkpoint and stats are saved') 
parser.add_argument('--vg_img_path',default='/home/aneesh/datasets/gqa_imgs/images/', 
                help='path to image directory')
parser.add_argument('--gqa_ann_path',default='/home/aneesh/datasets/gqa_ann/OpenSource/', 
                help='path to annotations')
parser.add_argument('--gqa_split_type',default='balanced', 
        help='GQA split eg: balanced , all')
# Segmentation #############################
# No idea what this sands for 
############################################
parser.add_argument(
        "--mask_model",
        default="none",
        type=str,
        choices=("none", "smallconv", "v2"),
        help="Segmentation head to be used (if None, segmentation will not be trained)",
    )
parser.add_argument('--masks',action='store_true') 
#endregion

#region: training hyperparameters: 
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--imset', default='train', type=str, metavar='IS',
                    help='train, val or test set')
parser.add_argument('--batch_size', default=2, type=int, metavar='n',
                    help='mini-batch size')
parser.add_argument('--learning_rate', default=1e-5, type=float, metavar='LR',
                    help='base learning rate')
parser.add_argument('--dropout', default=0.01, type=float, metavar='d',
                    help='dropout for training translation transformer')
##################################################
parser.add_argument('--weight_decay', default=0.5, type=float, metavar='W',
                    help='weight decay')
##################################################
parser.add_argument('--momentum', default=0.95, type=float, metavar='M',
                    help='momentum for sgd')
parser.add_argument('--clip', default=100, type=float, metavar='GC',
                    help='Gradient Clipping')
parser.add_argument('--betas', default=(0.9, 0.98), type=tuple, metavar='B',
                    help='betas for Adam Optimizer')
parser.add_argument('--eps', default=1e-9, type=float, metavar='E',
                    help='eps for Adam optimizer')
parser.add_argument('--loss_fn', default='cross_entropy', type=str, metavar='LF',
                    help='loss function for translation')
##################################################################
parser.add_argument('--lambd', default=0.5, type=float, metavar='L',
                    help='lambd of loss function')
##################################################################
parser.add_argument('--optimizer', default='adam', type=str, metavar='OP',
                    help='selecting optimizer')
#endregion

#region: slot-attention hyperparameters: 
parser.add_argument('--simg', default=15, type=int, metavar='SI',
                    help='number of slots for image modality')
parser.add_argument('--itersimg', default=5, type=int, metavar='II',
                    help='numer of iterations for slot attention on images')
parser.add_argument('--slotdimimg', default=768, type=int, metavar='SDI',
                    help='dimension of slots for images')
parser.add_argument('--stext', default=15, type=int, metavar='ST',
                    help='number of slots for text modality')
parser.add_argument('--iterstext', default=4, type=int, metavar='IT',
                    help='number of iterations for slot attention on text')
parser.add_argument('--slotdimtext', default=512, type=int, metavar='IT',
                    help='number of iterations for slot attention on text')
#endregion

#region: transformer encoder hyperparameters: 
parser.add_argument('--nhead', default=8, type=int, metavar='NH',
                    help='number of heads in transformer')
parser.add_argument('--tdim', default=512, type=int, metavar='D',
                    help='dimension of transformer')
parser.add_argument('--nlayers', default=3, type=int, metavar='NL',
                    help='number of layers in transformer')
#endregion

#region: tokenizer
parser.add_argument('--text_encoder_type', default='openai/clip-vit-base-patch32', type=str, metavar='T',
                    help='text encoder')
#endregion


args = parser.parse_args()
#endregion


class Clip_feat_extractor(nn.Module): 
    def __init__(self, processor): 
        super().__init__()
        self.processor = processor

    def __call__(self, img): 
        return self.processor(img)['pixel_values'][0]

class Transf_CLIProcess(nn.Module): 
    def __init__(self, processor): 
        super().__init__()
        self.processor = processor
    
    def __call__(self, image): 
        return self.processor(images=image, return_tensors='pt')['pixel_values'] 

########################################
## changed outer torch.sum to torch.mean
########################################
def min_l2_loss(inp, trg): 
    return torch.mean(torch.min(torch.mean((inp-trg.unsqueeze(1))*(inp-trg.unsqueeze(1)), axis=(2,3,4)), axis=1).values)

def l2_loss(inp, trg): 
    return torch.mean((inp-trg)*(inp-trg))

def tensor2img(img:torch.tensor): 
    #inp.shape = 3, H, W

    test_img = img.permute(1,2,0)
    test_img = test_img.cpu().detach().numpy()
    test_img = 255*test_img
    test_img = test_img.astype(np.uint8)
    return test_img


#region: MAIN FUNCTION
def main(): 

    # single-node distributed training
    args.rank = 0
    args.dist_url = 'tcp://localhost:58472'
    args.world_size = 1 
    ############################################
    args.ngpus_per_node = args.world_size #single machine 
    ############################################
    # print(args.ngpus_per_node)
    # torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)
    torch.multiprocessing.spawn(main_worker, args=(args,), nprocs=args.ngpus_per_node)
#endregion


def main_worker(gpu, args):
    
    args.rank += gpu
    # print(args.rank, args.world_size)
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    if args.rank == 0:

        
        wandb.init(config=args, project='slot_vqa')#############################################
        wandb.run.name = f"pc-dry-run-{wandb.run.id}"
        wandb.config.update(args)
        config = wandb.config
        wandb.run.save()

        # exit()
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)
        

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    # clip_feat = CLIPFeatureExtractor(do_resize=False, do_center_crop=False)
    # clip_feat_extractor = Clip_feat_extractor(clip_feat)
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    dataset = build(root='/home/aneesh/datasets/PhraseCutDataset/data/VGPhraseCut_v0', 
    sett='val',
    resolution=[224, 224], 
    transform= [Transf_CLIProcess(processor)])
    #initializing the model: 
    mbert = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32').to(args.rank)
    # mbert = BertModel.from_pretrained('bert-base-multilingual-uncased').to(args.rank)
    for param in mbert.parameters(): 
        param.requires_grad=False

    clip_vision_model = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32').to(args.rank)
    for param in clip_vision_model.parameters(): 
        param.requires_grad=False

    model = SlotVQA(clip_vision_model, mbert, 
                resolution=(224, 224), 
                slots_img=args.simg, iters_img=args.itersimg, slot_dim_img=args.slotdimimg, 
                slots_text=args.stext, iters_text=args.iterstext, slot_dim_text=args.slotdimtext, 
                num_head=args.nhead, transf_dim=args.tdim, transf_num_layers=args.nlayers).to(args.rank)
    
    #wrapping the model in DistributedDataParallel 
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)

    #defining the optimizer
    if args.optimizer == 'adam':
        optimizer =torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=args.betas, eps=args.eps) 
    else: 
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay) 


    #defining sampler
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    
    print('instantiating dataloader')
    loader = torch.utils.data.DataLoader(
         dataset, batch_size=per_device_batch_size, num_workers=args.workers,
         pin_memory=True, sampler=sampler, collate_fn = MyCollate(tokenizer=dataset.tokenizer))
    
    # val_loader = torch.utils.data.DataLoader(
    #      val_dataset, batch_size=per_device_batch_size, num_workers=args.workers,
    #      pin_memory=True, sampler=val_sampler, collate_fn = MyCollate(tokenizer=dataset.tokenizer))


    #training loop: 
    start_epoch = 0 
    ###################################
    #can edit this to start from saved epoch
    best_accuracy=0
    for epoch in range(start_epoch, args.epochs): 


        start_time = time.time()
        model.train()
        
        ## freeze weights of network 
        ## finetune decoder
        if epoch>10: 
            model.eval()
            model.decoder.train()

        sampler.set_epoch(epoch)
        epoch_loss = 0 
        acc=0
        
        counter = 0 
        for step, item in enumerate(loader, start=epoch*len(loader)): 
            
            model.train() 
            img = item[0].cuda(gpu, non_blocking=True)
            phrase = item[1].cuda(gpu, non_blocking=True)
            trg_mask = item[2].cuda(gpu, non_blocking=True)

            recon_combined, recons, masks = model(img, phrase, object_seg=True)

            # print(f'img.shape: {img.shape}')
            # print(f'phrase.shape: {phrase.shape}')
            # print(f'trg_mask.shape: {trg_mask.shape}')
            # img_save = tensor2img(img[0])
            # Image.fromarray(img_save).convert("RGB").save("img.png")
            # mask_save = tensor2img(trg_mask[0])
            # Image.fromarray(mask_save).convert("RGB").save("trg_mask.png")
            # test_mask = plt.matshow(trg_mask[0].detach().cpu())
            # test_recon = plt.matshow(recon_combined[0].detach().cpu())
 
            optimizer.zero_grad()

            trg_obj = img*trg_mask

            # print(f'recon_combined.shape: {recon_combined.shape}')
            # print(f'recons.shape: {recons.shape}')
            # print(f'masks.shape: {masks.shape}')
            #find the shapes, save the image
            # print(f'type:{type(min_l2_loss(recons,trg_obj))}, {type(l2_loss(recon_combined,img))}')
            loss = min_l2_loss(recons,trg_obj) + args.lambd*l2_loss(recon_combined, img)
            wandb.log({'iter_loss': loss})
            # print(loss)
            
            loss.backward()
            
            optimizer.step()
            counter+=1
            epoch_loss+=loss.item()
        
        if args.rank==0: 
            wandb.log({'epoch_loss':epoch_loss/counter})
            if epoch%5==0:
                with torch.no_grad(): 
                    img_save = tensor2img(img[0])
                    caption = [dataset.tokenizer.convert_ids_to_tokens(i.item()) for i in phrase[0]] 
                    img_save = wandb.Image(img_save, caption=caption)
                    wandb.log({'image': img_save})
                    mask_save = tensor2img(trg_obj[0])
                    mask_save = wandb.Image(mask_save)
                    wandb.log({'trg_obj': mask_save})
                    recons_save = tensor2img(recon_combined[0])
                    recons_save = wandb.Image(recons_save)
                    wandb.log({'pred': recons_save})

if __name__ == '__main__': 
    main()
