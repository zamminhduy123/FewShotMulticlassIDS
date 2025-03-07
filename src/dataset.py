

# coding=utf-8
from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import numpy as np
import shutil
import errno
import torch
import math
import os
from utils import create_prototypes_and_queries_with_time

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

class RoadDataset_v3(data.Dataset):
    on_road = False
    road_type = 'fab_raw_split'

    # Change these paths to the location of the dataset on your machine
    road_path = '/home/ntmduy/CANET/CICIDS2017/data/road/'
    can_train_and_test_path = '/home/ntmduy/can-train-and-test/set_01/processed/'
    ch_path = '/home/ntmduy/CANET/CICIDS2017/data/car-hacking/'
    
    def __init__(self, dataset=0, mode='train', road_type="fab_raw_split", with_time=False, embed = 12, seperate_proto=False, shot=5, new=False, ws = 15, s = 15):
        super(RoadDataset_v3, self).__init__()
        self.on_road = dataset == 1
        folder = 'new-road' if new and self.on_road else 'road' if self.on_road else 'car-hacking' if (dataset == 0) else 'can_train_and_test'

        self.ws = ws
        s = s
        self.road_type = road_type
        self.with_time = with_time

        path = '/home/ntmduy/CANET/CICIDS2017/data/new-road/'if new and self.on_road else self.road_path if self.on_road else self.ch_path if (dataset == 0) else self.can_train_and_test_path

        # can_train_and_test
        #/home/ntmduy/can-train-and-test/set_01/processed/timete-set_01-2d-dec-can_train_and_test_1_ss_3_no-16-16.npy
        
        if (mode == 'train'):
            #xtrain-road-2d-dec-fab_ss_11_no_split_fuzzing_90test-16-1.npy
            #/home/ntmduy/CANET/CICIDS2017/data/road/timetr-road-2d-dec-fab_ss_11_2d_no_split_90test_fuzzing-16-1.npy
            
            #road-2d-dec-fab_ss_11_2d_no_split_90test_fuzzing-16-1.npy
            self.x = np.load(os.path.join(path, f'xtrain-{folder}-2d-dec-{self.road_type}-{self.ws}-{s}.npy')).astype(np.float32)
            self.y = np.load(os.path.join(path, f'ytrain-{folder}-2d-dec-{self.road_type}-{self.ws}-{s}.npy')).astype(np.int64)  
            if (self.with_time):
                self.time = np.load(os.path.join(path, f'timetr-{folder}-2d-dec-{self.road_type}-{self.ws}-{s}.npy')).astype(np.float32)  

            # Select limit samples from each class
            # unique_classes = np.unique(self.y)
            # selected_indices = []
            # limit = 4000
            # for cls in unique_classes:
            #     cls_indices = np.where(self.y == cls)[0]
            #     if len(cls_indices) <= limit:
            #         selected_indices.extend(cls_indices)
            #     else:
            #         selected_indices.extend(np.random.choice(cls_indices, size=limit, replace=False))
            
            # self.x = self.x[selected_indices]
            # self.y = self.y[selected_indices]
            # self.time = self.time[selected_indices]

        elif (mode == 'val'):
            if (self.with_time):
                self.time = np.load(os.path.join(path, f'timete-{folder}-2d-dec-{self.road_type}-{self.ws}-{s}.npy')).astype(np.float32) 
            self.x = np.load(os.path.join(path, f'xval-{folder}-2d-dec-{self.road_type}-{self.ws}-{s}.npy')).astype(np.float32)
            self.y = np.load(os.path.join(path, f'yval-{folder}-2d-dec-{self.road_type}-{self.ws}-{s}.npy')).astype(np.int64)
        elif (mode == 'test'):
            if (self.with_time):
                self.time = np.load(os.path.join(path, f'timete-{folder}-2d-dec-{self.road_type}-{self.ws}-{s}.npy')).astype(np.float32) 
            self.x = np.load(os.path.join(path, f'xtest-{folder}-2d-dec-{self.road_type}-{self.ws}-{s}.npy')).astype(np.float32)
            self.y = np.load(os.path.join(path, f'ytest-{folder}-2d-dec-{self.road_type}-{self.ws}-{s}.npy')).astype(np.int64)
        elif (mode == 'unknown'):
            if (dataset == 1):
                self.x = np.load(os.path.join(path, f'fuzzing_ss_11_2d_no_data.npy')).astype(np.float32)
                self.y = np.load(os.path.join(path, f'fuzzing_ss_11_2d_no_label.npy')).astype(np.int64)
            elif (dataset == 0):
                #/home/ntmduy/CANET/CICIDS2017/data/car-hacking/DoS_ss_11_2d_no_data.npy
                self.x = np.load(os.path.join(path, f'DoS_ss_11_2d_no_data.npy')).astype(np.float32)
                self.y = np.load(os.path.join(path, f'DoS_ss_11_2d_no_label.npy')).astype(np.int64)
        
        self.x = np.nan_to_num(self.x)
        if (self.with_time):
            self.time = np.nan_to_num(self.time)
        self.y = np.squeeze(self.y)

        #TIME CONFIG
        self.embed = embed
        self.max_time_position = 10000
        self.gran = 1e-7 # ori: 1e-6
        self.log_e = 2
        self.d_model = 12

        self.pe = torch.tensor(
            [[pos / (10000.0 ** (i // 2 * 2.0 / self.d_model)) for i in range(self.embed)] for pos in
            range(self.max_time_position)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])  # Use sin for even columns
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])  # Use cos for odd columns

        self.seperate_proto = seperate_proto
        # Seperate the dataset into support set and query set
        if (seperate_proto):
            self.proto_x, self.proto_y, self.proto_time, self.query_x, self.query_y, self.query_time = create_prototypes_and_queries_with_time(
                torch.tensor(self.x), torch.tensor(self.y), torch.tensor(self.time), M=len(np.unique(self.y)), K=shot)
            print("M", len(np.unique(self.y)), "Way", "K", shot, "Shot")
            self.query_x, self.query_y, self.query_time = self.query_x.cpu().numpy(), self.query_y.cpu().numpy(), self.query_time.cpu().numpy()
            
            print("dataset", np.shape(self.proto_x), np.shape(self.proto_y), np.shape(self.x), np.shape(self.y), np.unique(self.y, return_counts=True))
            return
        print("dataset", np.shape(self.x), np.shape(self.y), np.unique(self.y, return_counts=True))

    def get_time(self, time_position):
        # Segment the corresponding position code according to the time position
        pe = torch.index_select(self.pe, 0, time_position)
        return pe

    def __getitem__(self, idx):
        x = self.x[idx] if not self.seperate_proto else self.query_x[idx]
        y = self.y[idx] if not self.seperate_proto else self.query_y[idx]
        if (self.with_time):
            time = self.time[idx] if not self.seperate_proto else self.query_time[idx]
        
            timestamp = np.ndarray.flatten(time)
            len_timestamp = len(time)
            for i in range(len_timestamp):
                ts = timestamp[i] / self.gran
                value = round(math.log(ts + 1, self.log_e))
                timestamp[i] = value
            for j in range(self.ws - len_timestamp):
                timestamp = np.append(timestamp, timestamp[len_timestamp - 1])

            time_feature = self.get_time(torch.IntTensor(timestamp))

        if (self.with_time):
            return torch.tensor(x), torch.tensor(y), time_feature
        return torch.tensor(x), torch.tensor(y)
    
    def get_proto_time(self, time):
        timestamp = np.ndarray.flatten(time)
        len_timestamp = len(timestamp)

        for i in range(len_timestamp):
            ts = timestamp[i] / self.gran
            value = round(math.log(ts + 1, self.log_e))
            timestamp[i] = value
        for j in range(self.ws - len_timestamp):
            timestamp = np.append(timestamp, timestamp[len_timestamp - 1])

        time_feature = self.get_time(torch.IntTensor(timestamp))

        return time_feature

    def __len__(self):
        return len(self.x) if not self.seperate_proto else len(self.query_x)
    
    def get_labels(self):
        return self.y
    