import random
import torch
import pandas as pd
from pathlib import Path

import torch.utils.data as data
from torch.utils.data import dataloader


class PandaData(data.Dataset):
    def __init__(self, dataset_cfg=None,
                 state=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg

        #---->data and label
        self.nfolds = self.dataset_cfg.nfold
        self.fold = self.dataset_cfg.fold
        self.feature_dir = self.dataset_cfg.data_dir
        self.csv_dir = self.dataset_cfg.label_dir    ### set this to PANDA_train_val.csv 
        self.slide_data = pd.read_csv(self.csv_dir)

        #---->order
        self.shuffle = self.dataset_cfg.data_shuffle

        #---->split dataset
        if state == 'train':
            self.slide_data = self.slide_data[self.slide_data['set'] == 'train']
            self.data = self.slide_data.loc[:, 'image_id'].dropna().reindex()
            self.label = self.slide_data.loc[:, 'isup_grade'].dropna().reindex()
            
        if state == 'val':
            self.slide_data = self.slide_data[self.slide_data['set'] == 'val']
            self.data = self.slide_data.loc[:, 'image_id'].dropna()
            self.label = self.slide_data.loc[:, 'isup_grade'].dropna()
            
        if state == 'test':
            self.slide_data = self.slide_data[self.slide_data['set'] == 'test']
            self.data = self.slide_data.loc[:, 'image_id'].dropna()
            self.label = self.slide_data.loc[:, 'isup_grade'].dropna()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        print(len(self.data))
        print(idx)
        slide_id = self.data[idx]
        label = int(self.label[idx])
        full_path = Path(self.feature_dir) / f'{slide_id}.pt'
        features = torch.load(full_path)

        #----> shuffle
        if self.shuffle == True:
            index = [x for x in range(features.shape[0])]
            random.shuffle(index)
            features = features[index]


        return features, label

