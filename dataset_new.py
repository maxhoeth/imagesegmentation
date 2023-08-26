import cv2
import numpy as np
from multiprocessing import Pool
import multiprocessing as mp
from functools import partial
from torch.utils.data import Dataset
import os
import torch


def get_filename(root_dir, folder):
    sample_path = root_dir + 'images/' + folder + '/'
    label_path = root_dir + 'gtFine/' + folder + '/'
    cities = os.listdir(sample_path)
    cities = [i for i in cities if '.' not in i]
    sample_list = []
    label_list = []
    for i in range(len(cities)):
        sample_list.append([sample_path + cities[i] + '/' + s for s in list(os.listdir(sample_path + cities[i]))])
        label_list.append([label_path + cities[i] + '/' + s for s in list(os.listdir(label_path + cities[i])) if 'labelTrainIds' in s])

        
    label_list = sorted(np.concatenate(label_list))
    sample_list = sorted(np.concatenate(sample_list))

    return sample_list, label_list

def encode_segmap(mask):
    # 255 -> 0
    mask[mask == 255] = 0
    # -1 -> 0
    mask[mask == -1] = 0
    return mask

def normalize(input_image):
    input_image = input_image / 255.0
    return input_image

def load_image(sample_path, mask_path, size=256):
    SIZE = size
    sample_file = cv2.imread(sample_path)
    mask_file = cv2.imread(mask_path)[:,:,0]
    image = cv2.resize(sample_file, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask_file, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)

    image = normalize(image)
    mask = encode_segmap(mask)
    image, mask = transform(image, mask)

    image = np.reshape(image, (3, SIZE, SIZE)).astype(np.float32)
    mask = np.reshape(mask, (SIZE, SIZE)).astype(np.float32)

    return image, mask

def transform(sample, label):
    if np.random.random() < 0.5:
        sample = cv2.flip(sample, 1)
        label = cv2.flip(label, 1)
    return sample, label


class Cityspaces(Dataset):
    def __init__(self, root_dir, folder="train", size=256):
        self.root_dir = root_dir
        sample_files, label_files = get_filename(self.root_dir, folder)
        self.files = []
        self.size = size
        for file in range(len(sample_files)):
            sample = {}
            sample['path'] = sample_files[file]
            sample['label'] = label_files[file]
            self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = self.files[idx]
        image, mask = self.load_image_multiprocessing(sample['path'], sample['label'])
        return image, mask
    
    def load_image_multiprocessing(self, image_path, label_path):
        # ...

        with mp.Pool() as pool:
            image,mask = pool.apply(self.load_image, args=(image_path, label_path))
            #mask = pool.apply(self.load_image, args=(label_path,))
    
        return image, mask
    
    @staticmethod
    def load_image(*args):
        return load_image(*args)

    @staticmethod
    def encode_segmap(mask):
        return encode_segmap(mask)

    @staticmethod
    def normalize(input_image):
        return normalize(input_image)

    @staticmethod
    def transform(sample, label):
        return transform(sample, label)
    
    
class Cityspaces2(Dataset):
    
    def __init__(self, root_dir, folder="train", size=256):
        self.root_dir = root_dir
        sample_files, label_files = get_filename(self.root_dir, folder)
        self.files = []
        self.size = size
        for file in range(len(sample_files)):
            sample ={}
            sample['path'] = sample_files[file]
            sample['label'] = label_files[file]
            self.files.append(sample)

    
    def __encode_segmap__(self, mask):
        #255 -> 0
        mask[mask == 255] = 0
        #-1 -> 0
        mask[mask == -1] = 0
        return mask

    def __normalize__(self, input_image):
        input_image = input_image / 255.0
        return input_image

    def __load_image__(self, sample_path, mask_path):
        SIZE = self.size
        sample_file = cv2.imread(sample_path)
        mask_file = cv2.imread(mask_path)[:,:,0]
        image = cv2.resize(sample_file, (SIZE, SIZE), interpolation = cv2.INTER_NEAREST)
        mask = cv2.resize(mask_file, (SIZE, SIZE), interpolation = cv2.INTER_NEAREST)

        
        image = self.__normalize__(image)
        mask = self.__encode_segmap__(mask)
        image, mask = self.__transform__(image, mask)
        
        image = np.reshape(image, (3, SIZE, SIZE)).astype(np.float32)
        mask = np.reshape(mask, (SIZE, SIZE)).astype(np.float32)
                
        return image, mask
    
    def __len__(self):
        return len(self.files)
    
    def __transform__(self, sample, label):
        if np.random.random() < 0.5:
            sample = cv2.flip(sample, 1)
            label = cv2.flip(label, 1)
        return sample, label


    def __getitem__(self, idx):
        sample = self.files[idx]
        image, mask = self.__load_image__(sample['path'], sample['label'])
        return torch.Tensor(image), torch.Tensor(mask)
