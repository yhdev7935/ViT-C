import torch
from .utils import glob_map

from cv2 import imread, resize

import random
import numpy as np 
import json

class LabelMapping:
    def __init__(self):
        self.mapping   = {}
        self.negatives = [] #list of keys that would indicate negative results

    def __len__(self):
        return len(set([self.mapping[key] for key in self.mapping]))

    def __str__(self):
        return str([key for key in self.mapping])
        
    def __repr__(self):
        return repr(self.mapping)

    def __getitem__(self, key):
        return self.mapping[key]

    def __setitem__(self, key, value):
        self.mapping[key] = value

    def __iter__(self):
        for key in self.mapping:
            yield key

    def _get_labels():
        return set([self.mapping[key] for key in self.mapping])

    def load(self, file):
        with open(file, "r") as f:
            data = json.load(f)

            for keys_to_label in data["mapping"]:
                keys  = keys_to_label["keys"]
                label = keys_to_label["label"]

                for key in keys:
                    self.mapping[key] = label

            self.negatives = data["negatives"]

    def save(self, file):
        mapping_set = {}
        for key in self.mapping:
            
            if self.mapping[key] in mapping_set:
                mapping_set[self.mapping[key]].append(key) #append key to list mapped by label
            else:
                mapping_set[self.mapping[key]] = [key]


        mapping_list = [{"keys" : mapping_set[label], "label" : label} for label in mapping_set]

        with open(file, "w") as f:
            json.dump({
                "mapping" : mapping_list,
                "negatives" : self.negatives
                }, f, indent=4)



class PlantDataset(torch.utils.data.Dataset):
    
    def __init__(self, root_dir = "./", path_pattern = "*", transform = None, cache_size = -1, label_map : LabelMapping = None):

        

        self.label_map    = label_map
        self.glob_mapping = glob_map(root_dir, path_pattern)
        self.root_dir     = root_dir
        self.transform    = transform
        self.cache      = {}
        self.cache_size = cache_size 
        # cache size essentially determines the size of the memory used to store images in memory
        # if cache size is -1, this essentially puts no limit on the images left in memory
        # if cache size is 1024, this means there's at most 1024 images stored in memory without referencing back to disk
        # since sampling is random, performance of cache is non-ideal, but still workable. 

        if len(self.glob_mapping) == 0:
            print("ERROR: Files not found. ")
            raise Exception
    
    def _get_labels(self):
        return [key for key in self.glob_mapping]
    def _get_num_labels(self):
        return len(self.glob_mapping)
    
    def __len__(self):
        return sum([len(self.glob_mapping[key]) for key in self.glob_mapping])
    
    def __getitem__(self, index):
        
        curr_index = 0
        image = None
        label_index = 0

        if index not in self.cache:
            for dir_index, dir_label in enumerate(self.glob_mapping):
                
                subset = self.glob_mapping[dir_label]
                
                if index < curr_index + len(subset):
                    image = resize(imread(str(subset[index - curr_index])), (256, 256))
                    
                    label_index = dir_index
                    label       = dir_label
                    break 
                else:
                    curr_index += len(subset)
            
            if image is IndexError:
                raise Exception

            if self.cache_size != -1 and len(self.cache) > self.cache_size:
                self.cache.pop(random.choice([key for key in self.cache]))

            if self.label_map:
                label_index = self.label_map[label]
            self.cache[index] = (image, label_index)
        else:
            image, label_index = self.cache[index]
        
        
        image = self.transform(image) if self.transform else image

        

        return image, label_index

# class TomatoDataset(torch.utils.data.Dataset):
#     """Generic tomato dataset"""

#     def __init__(self, root_dir = "./", transform = None):
        
#         self.glob_mapping = glob_map(root_dir, "Tomato*")
#         self.root_dir     = root_dir
#         self.transform    = transform

#         if len(self.glob_mapping) == 0:
#             print("ERROR: Files not found. ")
#             raise Exception

#     def _get_labels(self):
#         return [key for key in self.glob_mapping]
#     def _get_num_labels(self):
#         return len(self.glob_mapping)
    
#     def __len__(self):
#         return sum([len(self.glob_mapping[key]) for key in self.glob_mapping])
    
#     def __getitem__(self, index):
        
#         curr_index = 0
#         image = None
#         label = 0
#         for dir_index, dir_label in enumerate(self.glob_mapping):
            
#             subset = self.glob_mapping[dir_label]
            
#             if index < curr_index + len(subset):
#                 image = imread(str(subset[np.abs(index - curr_index)]))
#                 label = dir_index
#                 break 
#             else:
#                 curr_index += len(subset)
        
#         if image is IndexError:
#             raise Exception
#         elif image is None:
#             print("Found None image. Defaulting...")
#             print(index)
#             print(curr_index + len(subset))
#             print(dir_label)
#             raise Exception

#         image = self.transform(image) if self.transform else image
#         return image, label
