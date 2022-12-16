
#################################################
# from easy_vqa import get_train_questions, get_test_questions
# from easy_vqa import get_train_image_paths, get_test_image_paths
# from datasets.test_easy_vqa import build, MyCollate
# from torchvision import transforms as T 
#################################################

#region: IMPORTS:
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

import torch
from torch import nn, optim 
import torch.nn.functional as F 
#from torchvision import transforms
from transformers import AutoTokenizer, BertModel, CLIPVisionModel, CLIPTextModel, CLIPTokenizer, CLIPProcessor

#from vg_dataloader import VG_dataset
###########
from datasets.gqa_tweaked import MyCollate
from datasets.gqa_tweaked import build 
##########
from model.model import SlotVQA

import wandb
import wandb.apis.reports as wr
import  pytorch_warmup as warmup
#endregion

#region: SETTING ENV VARIABLES: 
os.environ['TRANSFORMERS_OFFLINE'] = 'yes'
os.environ['WANDB_START_METHOD'] = 'thread'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
#endregion

#region: SETTING SEED: 
MANUAL_SEED = 3407

random.seed(MANUAL_SEED)
np.random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)
torch.backends.cudnn.deterministic = True
#endregion

#region: ARGUMENTS:
parser = argparse.ArgumentParser(description='slot_vqa')

#system config: 
parser.add_argument('--workers', default=5, type=int, metavar='N', 
                    help='number of data loader workers') 
parser.add_argument('--print_freq', default=2, type=int, metavar='PF', 
                    help='write in the stats file and print after PF steps') 
parser.add_argument('--load', default=False , type=bool, metavar='L', 
                    help='load pretrained model if True') 
parser.add_argument('--checkpoint_dir', default='/home/aneesh/vqa_checkpoint/checkpoint/', type=Path, metavar='CD', 
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
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--imset', default='train', type=str, metavar='IS',
                    help='train, val or test set')
parser.add_argument('--batch_size', default=64, type=int, metavar='n',
                    help='mini-batch size')
parser.add_argument('--learning_rate', default=0.001, type=float, metavar='LR',
                    help='base learning rate')
parser.add_argument('--dropout', default=0.1, type=float, metavar='d',
                    help='dropout for training translation transformer')
parser.add_argument('--weight_decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
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
parser.add_argument('--optimizer', default='sgd', type=str, metavar='OP',
                    help='selecting optimizer')

#slot-attention hyperparameters: 
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

#transformer encoder hyperparameters: 
parser.add_argument('--nhead', default=8, type=int, metavar='NH',
                    help='number of heads in transformer')
parser.add_argument('--tdim', default=512, type=int, metavar='D',
                    help='dimension of transformer')
parser.add_argument('--nlayers', default=3, type=int, metavar='NL',
                    help='number of layers in transformer')

#tokenizer
parser.add_argument('--text_encoder_type', default='openai/clip-vit-base-patch32', type=str, metavar='T',
                    help='text encoder')


args = parser.parse_args()

if args.mask_model != "none" :
    args.masks = True
#endregion

#region: HELPER FUNCTIONS: 
class Transf_CLIProcess(nn.Module): 
    def __init__(self, processor): 
        super().__init__()
        self.processor = processor
    
    def __call__(self, image): 
        return self.processor(images=image, return_tensors='pt')['pixel_values'] 

#endregion

# region: WANDB REPORT: 
'''
report  = wr.Report(
        project="slot_vqa",
        title = "VQA Experiments",
        description= "Experiments with VQA-hyperparamter tuning and training."
) 
'''
#endregion
    

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    wandb.init(config=args, project='slot_vqa')#############################################
    wandb.run.name = f"gqa-no-slots-guide=text-emb"
    wandb.config.update(args)
    config = wandb.config
    wandb.run.save()

    # exit()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
    print(' '.join(sys.argv))
    print(' '.join(sys.argv), file=stats_file)
        


#loading dataset: 
    ###################################################3
    #########################################
    #GQA DATASET
    #########################################
    args.imset = 'train'
    dataset = build(image_set=args.imset, 
                    args=args)
    ans_dict_len = len(dataset.answer2id)#1853 including unk

    args.imset = 'val'
    val_dataset = build(image_set=args.imset, 
                        args=args)
    ###############################################
    #EASY DATASET  
    ################################################
    # train_questions, train_answers, train_image_ids = get_train_questions()
    # # print(train_image_ids)
    # train_image_ids = list(set(train_image_ids))
    # train_image_paths = get_train_image_paths()
    # tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')

    # processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    # transform = [T.Resize([224, 224]), Transf_CLIProcess(processor)]
    # # transform = [T.ToTensor(), T.Resize([400, 400])]

    # # print(train_answers)
    # dataset = build(train_questions, train_answers, train_image_ids, 
    #             train_image_paths, tokenizer, transform)
    # ans_dict_len = len(dataset.ans2id)
    # # print(dataset.ans2id)

    # test_questions, test_answers, test_image_ids = get_test_questions()
    # test_image_paths = get_test_image_paths()
    # val_dataset = build(test_questions, test_answers, test_image_ids, 
    #             test_image_paths, tokenizer, transform)


    ###############################################
    #COCO-QA DATASET  
    ################################################
    # processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    # transform = [T.Resize([224, 224]), Transf_CLIProcess(processor)]
    # # transform = [T.ToTensor(), T.Resize([400, 400])]
    # args.root = '/home/aneesh/datasets/'
    # dataset = build(transform, 
    #                 args=args)
    # ans_dict_len = len(dataset.answer2id)#1853 including unk

    # args.imset = 'test'
    # val_dataset = build(transform, 
    #                     args=args)

#initializing the model: 
    mbert = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
    # mbert = BertModel.from_pretrained('bert-base-multilingual-uncased').to(args.rank)
    for param in mbert.parameters(): 
        param.requires_grad=False

    clip_vision_model = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
    for param in clip_vision_model.parameters(): 
        param.requires_grad=False

    model = SlotVQA(clip_vision_model, mbert, 
                resolution=(224, 224), 
                slots_img=args.simg, iters_img=args.itersimg, slot_dim_img=args.slotdimimg, 
                slots_text=args.stext, iters_text=args.iterstext, slot_dim_text=args.slotdimtext, 
                num_head=args.nhead, transf_dim=args.tdim, transf_num_layers=args.nlayers, 
                ans_dim=ans_dict_len).to(device)

    if args.load:  
        ckpt = torch.load(args.checkpoint_dir/'checkpoint_best.pth', 
            map_location='cpu') 
    
        print(type(args.load), args.load)
        model.load_state_dict(ckpt['model'])
        print("Pretrained Model Loaded")
    
    model = model.to(device)
    
    #wrapping the model in DistributedDataParallel 
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]

    #defining the optimizer
    if args.optimizer == 'adam':
        optimizer =torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=args.betas, eps=args.eps) 
    else: 
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay) 
    
    #defining the loss_function: 
    loss_fn = nn.CrossEntropyLoss()


    # assert args.batch_size % args.world_size == 0
    # per_device_batch_size = args.batch_size // args.world_size
    
    print('instantiating dataloader')
    loader = torch.utils.data.DataLoader(
         dataset, batch_size=args.batch_size, num_workers=args.workers,
         pin_memory=True, #sampler=sampler, collate_fn = MyCollate(tokenizer=dataset.tokenizer))
         collate_fn = MyCollate(tokenizer=dataset.tokenizer))           
    
    val_loader = torch.utils.data.DataLoader(
         val_dataset, batch_size=args.batch_size, num_workers=args.workers,
         pin_memory=True, #sampler=val_sampler, collate_fn = MyCollate(tokenizer=dataset.tokenizer))
         collate_fn = MyCollate(tokenizer=dataset.tokenizer))

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(loader)*args.epochs)
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=10000)

    #training loop: 
    start_epoch = 0 
    ###################################
    #can edit this to start from saved epoch
    best_accuracy=0
    for epoch in range(start_epoch, args.epochs): 


        start_time = time.time()
        model.train()
        epoch_loss = 0 
        acc=0
        
        counter = 0 
        for step, item in enumerate(loader, start=epoch*len(loader)): 
            
            model.train() 
            img = item[0].to(device)#.cuda(gpu, non_blocking=True)
            ques = item[1].to(device)#.cuda(gpu, non_blocking=True)cuda(gpu, non_blocking=True)
            ans = item[2].to(device)#cuda(gpu, non_blocking=True)

            ####################################################
            # img = img*img.shape[0]
            ####################################################

            # print(f'img.shape={img.shape}, ques.shape={ques.shape} ,ans.shape={ans.shape}') 
                
            pred = model(img, ques)
            # print(pred)
            # print(f'pred = {pred_ans}')
            # print(f'ans={ans}')
            
            optimizer.zero_grad()

            # print(f'ans.shape={ans.shape}, pred.shape={pred.shape}')
            loss = loss_fn(pred, ans)
            wandb.log({"iter_loss": loss})
            # torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            loss.backward()

            optimizer.step()
            with warmup_scheduler.dampening():
                lr_scheduler.step()
            model.eval()
            pred_soft = F.softmax(pred, dim=1)
            pred_ans = torch.argmax( pred_soft,dim=1)

            counter += 1

            epoch_loss+=loss.item()

            # pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
            assert pred_ans.dtype == ans.dtype, f'Expected prediction and targets to be of same dype but got pred.dtype={pred.dtype} and ans.dtype={ans.dtype}'
            # device = pred_ans.device
            # print(f'pred===========================> {pred}')
            # print(f'target=============================> {ans}')

            correct = torch.zeros(pred_ans.shape).to(device).masked_fill(pred_ans==ans, 1)
            # print(correct)
            assert len(correct.shape) == 1, f'Expected predictions.shape == [n]  but got predictions.shape {correct.shape} instead'
            correct_ = torch.sum(correct)
            # torch.distributed.all_reduce(correct_)
            # print(correct)
            acc += correct_.div_(pred_ans.shape[0])
            # torch.distributed.all_reduce(acc)
            # print(acc)
        train_acc = acc.div_(counter)
        # torch.distributed.all_reduce(train_acc)
        train_acc = train_acc*100
        wandb.log({"epoch_loss": epoch_loss/counter})
        wandb.log({"train accuracy": train_acc})
        print({"train accuracy": train_acc})
        if epoch%args.print_freq==0: 
            
            #test the model: 
            acc = 0
        
            counter2=0
            for step2, item in enumerate(val_loader): 
                
                model.eval()
                
                img = item[0].to(device)#cuda(gpu, non_blocking=True)
                ques = item[1].to(device)#cuda(gpu, non_blocking=True)
                ans = item[2].to(device)#cuda(gpu, non_blocking=True)
 
                
                if step2 ==0: 
                    return_text_att = True
                    att, pred = model(img, ques, return_text_att)
                    # att_map = plt.matshow(att[0].detach().cpu())
                    # caption = [dataset.tokenizer.convert_ids_to_tokens(i.item()) for i in ques[1]] 
                    # caption = f'epoch:{epoch} question: {caption}'
                    # image = wandb.Image(att_map, caption=caption)
                    # wandb.log({'text_attention_map': image})
            
                else: 
                    return_text_att = False
                    pred = model(img, ques)


                # print(f'pred before softmax ====================> {pred}')
                pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
                assert pred.dtype == ans.dtype, f'Expected prediction and targets to be of same dype but got pred.dtype={pred.dtype} and ans.dtype={ans.dtype}'
                device = pred.device
                # print(f'pred===========================> {pred}')
                # print(f'target=============================> {ans}')
                correct = torch.zeros(pred.shape).to(device).masked_fill(pred==ans, 1)
                assert len(correct.shape) == 1, f'Expected predictions.shape == [n]  but got predictions.shape {correct.shape} instead'
                correct_ = torch.sum(correct)
                # torch.distributed.all_reduce(correct_)
                counter2+=1
                acc += correct_.div_(pred.shape[0])
                # torch.distributed.all_reduce(acc)

            accuracy = acc.div_(counter2)*100
            print(f'test_acc={accuracy}')
            # print(pred.shape[0])
            # accuracy = torch.distributed.all_reduce(accuracy)
            wandb.log({"accuracy": accuracy})
            print(accuracy)
            if accuracy> best_accuracy: 
                try: 
                    state = dict(epoch=epoch + 1, model=model.state_dict(),
                                optimizer=optimizer.state_dict())
                    torch.save(state, args.checkpoint_dir / f'checkpoint_best.pth')
                    print('Model saved in', args.checkpoint_dir)
                except:
                    print('failed to save model')
                best_accuracy= accuracy


                stats = dict(epoch=epoch, step=step, 
                            loss=loss.item(),acc=accuracy.item(),  
                            time=int(time.time() - start_time))            

                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)
            
                '''
                if epoch%5 == 0: 

                    state = dict(epoch=epoch + 1, model=model.module.state_dict(),
                                optimizer=optimizer.state_dict())
                    torch.save(state, args.checkpoint_dir / f'checkpoint_{accuracy}_{epoch+1}.pth')
                    print('Model saved in', args.checkpoint_dir)
                '''

if __name__ == '__main__': 
    main()
    wandb.finish()



