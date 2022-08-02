#################################################
from easy_vqa import get_train_questions, get_test_questions
from easy_vqa import get_train_image_paths, get_test_image_paths
from datasets.test_easy_vqa import build, MyCollate
from torchvision import transforms as T 
#################################################

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

import torch
from torch import nn, optim 
import torch.nn.functional as F 
#from torchvision import transforms
from transformers import AutoTokenizer, BertModel, CLIPVisionModel, CLIPTextModel, CLIPTokenizer, CLIPProcessor

#from vg_dataloader import VG_dataset
###########
'''
from datasets.gqa_tweaked import build, MyCollate
'''
##########
from model.model import SlotVQA

import wandb

# setting environment variables: 
os.environ['TRANSFORMERS_OFFLINE'] = 'yes'
os.environ['WANDB_START_METHOD'] = 'thread'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# setting seed: 
MANUAL_SEED = 4444

random.seed(MANUAL_SEED)
np.random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='slot_vqa')

#system config: 
parser.add_argument('--workers', default=5, type=int, metavar='N', 
                    help='number of data loader workers') 
parser.add_argument('--print_freq', default=1, type=int, metavar='PF', 
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

#training hyperparameters: 
parser.add_argument('--epochs', default=25, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--imset', default='train', type=str, metavar='IS',
                    help='train, val or test set')
parser.add_argument('--batch_size', default=8, type=int, metavar='n',
                    help='mini-batch size')
parser.add_argument('--learning-rate', default=0.2, type=float, metavar='LR',
                    help='base learning rate')
parser.add_argument('--dropout', default=0.01, type=float, metavar='d',
                    help='dropout for training translation transformer')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd')
parser.add_argument('--clip', default=1, type=float, metavar='GC',
                    help='Gradient Clipping')
parser.add_argument('--betas', default=(0.9, 0.98), type=tuple, metavar='B',
                    help='betas for Adam Optimizer')
parser.add_argument('--eps', default=1e-9, type=float, metavar='E',
                    help='eps for Adam optimizer')
parser.add_argument('--loss_fn', default='cross_entropy', type=str, metavar='LF',
                    help='loss function for translation')
parser.add_argument('--optimizer', default='adam', type=str, metavar='OP',
                    help='selecting optimizer')

#slot-attention hyperparameters: 
parser.add_argument('--simg', default=10, type=int, metavar='SI',
                    help='number of slots for image modality')
parser.add_argument('--itersimg', default=1, type=int, metavar='II',
                    help='numer of iterations for slot attention on images')
parser.add_argument('--slotdimimg', default=256, type=int, metavar='SDI',
                    help='dimension of slots for images')
parser.add_argument('--stext', default=7, type=int, metavar='ST',
                    help='number of slots for text modality')
parser.add_argument('--iterstext', default=5, type=int, metavar='IT',
                    help='number of iterations for slot attention on text')
parser.add_argument('--slotdimtext', default=256, type=int, metavar='IT',
                    help='number of iterations for slot attention on text')

#transformer encoder hyperparameters: 
parser.add_argument('--nhead', default=4, type=int, metavar='NH',
                    help='number of heads in transformer')
parser.add_argument('--tdim', default=256, type=int, metavar='D',
                    help='dimension of transformer')
parser.add_argument('--nlayers', default=3, type=int, metavar='NL',
                    help='number of layers in transformer')

#tokenizer
parser.add_argument('--text_encoder_type', default='openai/clip-vit-base-patch32', type=str, metavar='T',
                    help='text encoder')


args = parser.parse_args()

if args.mask_model != "none" :
    args.masks = True

class Transf_CLIProcess(nn.Module): 
    def __init__(self, processor): 
        super().__init__()
        self.processor = processor
    
    def __call__(self, image): 
        return self.processor(images=image, return_tensors='pt')['pixel_values'] 

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

def main_worker(gpu, args):
    
    args.rank += gpu
    # print(args.rank, args.world_size)
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    if args.rank == 0:

        wandb.init(config=args, project='slot_vqa')#############################################
        wandb.run.name = f"easy-vqa-{wandb.run.id}"
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

#loading dataset: 
    ###################################################3
    '''
    dataset = build(image_set=args.imset, 
                    args=args)
    ans_dict_len = len(dataset.answer2id)#1853 including unk

    args.imset = 'val'
    val_dataset = build(image_set=args.imset, 
                        args=args)
    '''
    ###############################################
    #EASY DATASET  
    ################################################
    train_questions, train_answers, train_image_ids = get_train_questions()
    train_image_paths = get_train_image_paths()
    # tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')

    # processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    # transform = [T.Resize([224, 224]), Transf_CLIProcess(processor)]
    transform = [T.ToTensor(), T.Resize([400, 400])]

    # print(train_answers)
    dataset = build(train_questions, train_answers, train_image_ids, 
                train_image_paths, tokenizer, transform)
    ans_dict_len = len(dataset.ans2id)
    # print(dataset.ans2id)

    test_questions, test_answers, test_image_ids = get_test_questions()
    test_image_paths = get_test_image_paths()
    val_dataset = build(test_questions, test_answers, test_image_ids, 
                test_image_paths, tokenizer, transform)

#initializing the model: 
    # mbert = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32').to(args.rank)
    mbert = BertModel.from_pretrained('bert-base-multilingual-uncased').to(args.rank)
    for param in mbert.parameters(): 
        param.requires_grad=False

    clip_vision_model = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32')
    for param in clip_vision_model.parameters(): 
        param.requires_grad=False

    model = SlotVQA(clip_vision_model, mbert, resolution=(600, 600), 
                slots_img=args.simg, iters_img=args.itersimg, slot_dim_img=args.slotdimimg, 
                slots_text=args.stext, iters_text=args.iterstext, slot_dim_text=args.slotdimtext, 
                num_head=args.nhead, transf_dim=args.tdim, transf_num_layers=args.nlayers, 
                ans_dim=ans_dict_len).to(args.rank)
    
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
    
    #defining the loss_function: 
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2)

    print('instantiated sampler')
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    
    print('instantiating dataloader')
    loader = torch.utils.data.DataLoader(
         dataset, batch_size=per_device_batch_size, num_workers=args.workers,
         pin_memory=True, sampler=sampler, collate_fn = MyCollate(tokenizer=dataset.tokenizer))
    
    val_loader = torch.utils.data.DataLoader(
         val_dataset, batch_size=per_device_batch_size, num_workers=args.workers,
         pin_memory=True, sampler=val_sampler, collate_fn = MyCollate(tokenizer=dataset.tokenizer))

    start_time = time.time()

    #training loop: 
    start_epoch = 0 
    ###################################
    #can edit this to start from saved epoch
    for epoch in range(start_epoch, args.epochs): 


        sampler.set_epoch(epoch)
        epoch_loss = 0 
        
        counter = 0 
        for step, item in enumerate(loader, start=epoch*len(loader)): 
            
            model.train()
            
            img = item[0].cuda(gpu, non_blocking=True)
            ques = item[1].cuda(gpu, non_blocking=True)
            ans = item[2].cuda(gpu, non_blocking=True)

            # print(f'img.shape={img.shape}, ques.shape={ques.shape} ,ans.shape={ans.shape}') 

            pred = model(img, ques)
            
            optimizer.zero_grad()

            # print(f'ans.shape={ans.shape}, pred.shape={pred.shape}')
            loss = loss_fn(pred, ans)
            wandb.log({"iter_loss": loss})
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            loss.backward

            optimizer.step()

            counter += 1

            epoch_loss+=loss.item()


            if args.rank==0: 
                wandb.log({"epoch_loss": epoch_loss/counter})
                if step%args.print_freq==0: 
                    
                    #test the model: 
                    acc = 0
                    val_sampler.set_epoch(epoch)
                
                    for step2, item in enumerate(val_loader): 
                        
                        model.eval()
                        
                        img = item[0].cuda(gpu, non_blocking=True)
                        ques = item[1].cuda(gpu, non_blocking=True)
                        ans = item[2].cuda(gpu, non_blocking=True)

                        # print(img[1]==img[0])
                        # print(ques[1]==ques[0]) 

                        pred = model(img, ques)

                        # print(f'pred before softmax ====================> {pred}')
                        pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
                        assert pred.dtype == ans.dtype, f'Expected prediction and targets to be of same dype but got pred.dtype={pred.dtype} and ans.dtype={ans.dtype}'
                        device = pred.device
                        # print(f'pred===========================> {pred}')
                        # print(f'target=============================> {ans}')
                        correct = torch.zeros(pred.shape).to(device).masked_fill(pred==ans, 0)
                        assert len(correct.shape) == 1, f'Expected predictions.shape == [n]  but got predictions.shape {correct.shape} instead'
                        correct = torch.sum(correct)
                        acc += correct/pred.shape[0]
                        wandb.log({"accuracy": acc})


                    stats = dict(epoch=epoch, step=step, 
                                loss=loss.item(),acc=acc.item(),  
                                time=int(time.time() - start_time))            

                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
            
            state = dict(epoch=epoch + 1, model=model.module.state_dict(),
                        optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / 'checkpoint.pth')
            print('Model saved in', args.checkpoint_dir)
            

if __name__ == '__main__': 
    main()
    wandb.finish()