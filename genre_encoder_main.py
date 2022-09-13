# 2022-09-10 13:54 Incheon

# --- import dataset ---
from utils.dataloader import mel_dataset
from torch.utils.data import DataLoader, random_split

# --- import train module ---
from train_module import TrainerModule

# --- import etc ---
from tqdm import tqdm
import os
import wandb
from utils.config_hook import yaml_config_hook
import numpy as np

# --- Define config ---
config_dir = os.path.join(os.path.expanduser('~'),'trainer_module/config')     
config = yaml_config_hook(os.path.join(config_dir, 'config.yaml'))


# --- collate batch for dataloader ---
def collate_batch(batch):
    x_train = [x for x, _ in batch]
    y_train = [y for _, y in batch]                  
        
    return np.array(x_train), np.array(y_train)

# --- top_n_genre---
def top_n_genre(dataset_dir, n):

    temp_list = []
    dataset_size_list = []
    sort_top_dataset = []

    for k in range(0,30):
        data = mel_dataset(dataset_dir, k)
        temp_list.append(data)
    for i in range(0,30):
        dataset_size_list.append(len(temp_list[i]))
    temp_dict = dict(zip(temp_list, dataset_size_list))

    sort_top_n = sorted(temp_dict.items(), key = lambda item: item[1], reverse=True)[:n]

    for i in range(len(sort_top_n)):
        sort_top_dataset.append(sort_top_n[i][0])

    return sort_top_dataset


if __name__ == '__main__':    
    
    dataset_dir = os.path.join(os.path.expanduser('~'), config['dataset_dir'])            
    trainer = TrainerModule(seed=42, config=config)

    print("Loading dataset...")
    
    dataset_list = top_n_genre(dataset_dir, 5)
    
    
    for index, data in enumerate(dataset_list):
        
        dataset_size = len(data)
        train_size = int(dataset_size * 0.8)    
        test_size = dataset_size - train_size

        train_dataset, test_dataset, = random_split(data, [train_size, test_size])



        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size']*8, shuffle=True, num_workers=0, collate_fn=collate_batch)
        test_dataloader = DataLoader(test_dataset, batch_size=int(config['batch_size']/4)*8, shuffle=True, num_workers=0, collate_fn=collate_batch)
        
        print(f"genre_encoder={index}")
        print(f"batch_size = {config['batch_size']}")
        print(f"learning rate = {config['learning_rate']}")
        print(f"train_size = {train_size}")
        print(f"test_size = {test_size}")
        print('Data load complete!\n')
        
        trainer.train_model(train_dataloader, test_dataloader,num_epochs=config['pretrain_epoch'])
        trainer.save_model(index, trainer.state.step)    
