import os 
from config_combined import hyperparameters 
# arguments for important hyperparameters: 

print(f'training VQA learning rate={hyperparameters["vqa"]["lr_vqa"]}, epochs={hyperparameters["epochs"]}')
os.system(f'python train.py --learning rate={hyperparameters["vqa"]["lr_vqa"]} --epochs={hyperparameters["vqa"]["epochs"]}')
print(f'training image segmentation: learning rate={hyperparameters["obj"]["lr_vqa"]}, epochs={hyperparameters["obj"]["epochs"]}')
os.system(f'python train_obj.py --learning_rate={hyperparameters["obj"]["lr_vqa"]} --epochs={hyperparameters["obj"]["epochs"]}')

