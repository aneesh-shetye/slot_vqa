# Experiments for VQA using slot attention
## Current architecture: 
![slot_exp_1](https://user-images.githubusercontent.com/63205095/171621058-b443f7c3-80ba-4644-812f-4085c024f291.png)

## Steps to run the code: 

### 1. Create a conda env: 

Create an environment: 
```
conda create --name slotvqa
```
Activate the environment: 
```
conda activate slotvqa
```
Install the dependancies: 
```
pip install -r requirements.txt
```

### 2. Downloading the data: 

#### GQA: 

### 3. Running the code: 

Log into wandb 
```
wandb login
```
(This will prompt a link to the wandb auth key. Copy and paste it in the terminal.)

Pre-training on GQA dataset: 

```
python train.py --epochs=35 
```

Training on PhraseCut dataset: 

```
python train_obj.py --epochs=10 --load=1 
```
Fine-tuning on GQA dataset: 

```
python train.py --epochs=25 --load=1
```
