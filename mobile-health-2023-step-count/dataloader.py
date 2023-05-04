# This code loads training data into 'npy' files for training and testing
import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import math
from Lilygo.Recording import Recording
from Lilygo.Dataset import Dataset
from os import listdir
from os.path import isfile, join
from math import sqrt
import numpy as np

class LilyGoDataset(Dataset):
    def __init__(self, data_dir, save_dir, device, type, packed=False, normalized=False) -> None:
        '''
        data_dir: Where "action_labels, downloads, pose_lists" directories locate.
        data_dir: Where "npy" files are saved.
        device: torch.device('cuda') or torch.device('cpu').
        type: Type of data, 'train' or 'val' or 'test'.
        packed: If True, directly load pre-packed '.npy' files, otherwise pack it.
        normalized: If True, the data is normalized.
        '''
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.device = device
        self.packed = packed
        self.type = type

        assert type in ['train', 'val', 'test']

        if packed:
            if type == 'train':
                self.data = np.load(self.save_dir + 'train_data.npy')
                self.data_count = np.load(self.save_dir + 'train_data_count.npy')
                self.activity_labels = np.load(self.save_dir + 'train_activity_label.npy')
                self.path_labels = np.load(self.save_dir + 'train_path_label.npy')
                self.loaction_labels = np.load(self.save_dir + 'train_loaction_label.npy')
            elif type == 'val':
                self.data = np.load(self.save_dir + 'val_data.npy')
                self.activity_labels = np.load(self.save_dir + 'val_activity_label.npy')
                self.path_labels = np.load(self.save_dir + 'val_path_label.npy')
                self.loaction_labels = np.load(self.save_dir + 'val_loaction_label.npy')
            elif type == 'test':
                self.data = np.load(self.save_dir + 'test_data.npy')
                self.activity_labels = np.load(self.save_dir + 'test_activity_label.npy')
                self.path_labels = np.load(self.save_dir + 'test_path_label.npy')
                self.loaction_labels = np.load(self.save_dir + 'test_loaction_label.npy')
        else:
            print(self.type + ' dataset has not been prepared. Preparing now.')
            self.data, data_count, self.activity_labels ,self.path_labels ,self.loaction_labels = self.load_data()
            print(self.type + ' data loading finished.')
            
    # data pre-processing
    # This function aims to find the component caused by gravity from data, which means the signal around 0 Hz
    def get_gravity(self, data):
        filtered_data = np.zeros_like(data)
        # Parameters in IIR filter
        alpla = [1, -1.979133761292768, 0.979521463540373]
        beta = [0.000086384997973502, 0.00012769995947004, 0.000086384997973502]
        # Formula of IIR filter
        for i in range(2, len(data)):
            filtered_data[i] = alpla[0] * (data[i] * beta[0] + data[i-1] * beta[1] + data[i-2] * beta[2] - filtered_data[i-1] * alpla[1] - filtered_data[i-2] * alpla[2])
        return filtered_data

    # This function aims to realize a low-pass filter with cutoff frequency = 1 Hz. Because according to massive amounts of data, the general 
    # minimum frequency of human walking is about 1 Hz
    def get_highpass(self, data):
        filtered_data = np.zeros_like(data)  # filtered_data
        alpla = [1, -1.905384612118461, 0.910092542787947]
        beta = [0.953986986993339, -1.907503180919730, 0.953986986993339]

        for i in range(2, len(data)):
            filtered_data[i] = alpla[0] * (data[i] * beta[0] + data[i-1] * beta[1] + data[i-2] * beta[2] - filtered_data[i-1] * alpla[1] - filtered_data[i-2] * alpla[2])
        return filtered_data

    # This funciton aims to realize a high-pass filter with cutoff frequency = 5 Hz. Because according to massive amounts of data, the general 
    # maximum frequency of human walking is about 5 Hz
    def get_lowpass(self, data):
        filtered_data = np.zeros_like(data)  # filtered_data
        alpla = [1, -1.80898117793047, 0.827224480562408]
        beta = [0.096665967120306, -0.172688631608676, 0.095465967120306]
        
        for i in range(2, len(data)):
            filtered_data[i] = alpla[0] * (data[i] * beta[0] + data[i-1] * beta[1] + data[i-2] * beta[2] - filtered_data[i-1] * alpla[1] - filtered_data[i-2] * alpla[2])
        return filtered_data
    # preprocess the signal to get more accurate results
    def pre_process(self, data):
        # Find the component caused by gravity from data and remove it from the singanl
        data_gravity = self.get_gravity(data)
        data_user = data - data_gravity
        # Get user's acceleration along the gravity direction by dot product
        data_acc = data_user * data_gravity
        # Add low pass and high pass filter to reduce noise in signal (possible human walking rate:1 - 5Hz)
        data_filtered = self.get_highpass(data_acc)
        data_filtered = self.get_lowpass(data_filtered)
        return data_filtered
    
    def one_hot(self, activity_labels):
        activitu_label_onehot = [0, 0, 0, 0]
        for i in range(4):
            if i in activity_labels:
                activitu_label_onehot[i] = 1
        return activitu_label_onehot
    
    def segment(self, raw_data, all_data, data_count):
        # Calculate window size
        sampling_rate = 200
        std_win = 3 #s
        window_size = round(std_win*sampling_rate)
        prev_data_count = len(all_data)
        for s in range(0, len(raw_data)-window_size, round(window_size/2)):
            all_data.append(raw_data[s:s+window_size])
        data_count.append(len(all_data)-prev_data_count)



    
    def load_data(self):
        data_folder = self.data_dir + self.type
        magn_data = []
        data_count = []
        activity_labels = []
        path_labels = []
        loaction_labels = []
        # data loaded from raw
        # Read in the raw sensor data from a folder
        filenames = [join(data_folder, f) for f in listdir(data_folder) if isfile(join(data_folder, f)) and f!='.DS_Store']
        filenames.sort()
        for i, filename in enumerate(filenames):
            trace = Recording(filename, no_labels=False, mute=True)
            # Calculate raw magnitude of accelerometer signal
            amagn_raw = [sqrt(a**2+trace.data['ay'].values[i]**2+trace.data['az'].values[i]**2)for i, a in enumerate(trace.data['ay'].values)]
            # Pre-process data
            amagn = self.pre_process(amagn_raw)
            self.segment(amagn, magn_data, data_count)
            # one hot encoding
            # activity_labels.append(self.one_hot(trace.labels['activities']))
            # path_labels.append(trace.labels['path_idx'])
            # loaction_labels.append(trace.labels['board_loc'])
            if i%10 == 0:
                print('%d/%d files loaded' % (i, len(filenames)))
        
        # Save the data and labels to a numpy file
        # np.save(self.save_dir + self.type+'_data.npy', np.array(magn_data))
        np.save(self.save_dir + self.type+'_data_count.npy', np.array(data_count))
        # np.save(self.save_dir + self.type+'_activity_label.npy', np.array(activity_labels))
        # np.save(self.save_dir + self.type+'_path_label.npy', np.array(path_labels))
        # np.save(self.save_dir + self.type+'_loaction_label.npy', np.array(loaction_labels))
        return magn_data, data_count, activity_labels ,path_labels ,loaction_labels







    



# testing code
data_dir = 'E:\\Sunzhichao\\ETHz\\2223Spring\\Mobile_Health\\data\\'
save_dir = '.\\Loaded_data\\'
dataset = LilyGoDataset(data_dir=data_dir, save_dir=save_dir, device='CPU', type='train', packed=True)
print(np.load(save_dir + 'train_data_count.npy'))

