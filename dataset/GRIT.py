import json
import os
import numpy as np
import pycocotools.mask as mask
import math
import skimage
import pickle
import requests
from tqdm import tqdm
from torch.utils.data import Dataset
import pandas as pd
import json

class GRITDataset(Dataset):
    # Path to instances.json
    # Path to COCO2014/COCO2017 train images
    def __init__(self, data_root, patch_size=16):
        super(GRITDataset, self).__init__()
        self.data_root = data_root
        self.patch_size=patch_size
        self.data = None
        for i in range(21):
            df = pd.read_parquet(os.path.join(data_root, f'coyo_{i}_snappy.parquet'), engine='pyarrow')
            if type(self.data) == None:
                self.data = df
            else:
                self.data = pd.concat([self.data, df])
                self.data.reset_index(drop=True, inplace=True)
        self.entries = self._load_dataset()
        del self.data
    def _load_dataset(self):
        self.entries = []
        print("Loading pretraining data ...")
        for i in tqdm(range(len(self.data))):
            
            entry = self.data.iloc[i, :]

            image_folder = str(i).zfill(9)[:5]
            path_to_image = os.path.join(self.data_root, "images", image_folder, str(i).zfill(9) + ".jpg")
            path_to_json = os.path.join(self.data_root, "images", image_folder, str(i).zfill(9) + ".json")

            if not os.path.exists(path_to_json):
                continue
            #print(path_to_json)
            try:
                with open(path_to_json, "r") as f:
                    json_data = json.load(f)
            except:
                print(f"Problem loading {path_to_json}")
            if json_data['status'] != 'success':
                continue
            # try:
       
            #     # pass the url into
            #     # request.hear
            #     response = requests.head(entry['url'])
                
            #     # check the status code
            #     if response.status_code == 200:
            #         pass
            #     else:
            #         continue
            # except requests.ConnectionError as e:
            #     continue

            for chunk in entry['ref_exps']:
                info = {}
                info['height'] = entry['height']
                info['width'] = entry['width']
                info['chunk'] = chunk
                if entry['width'] < 16 or entry['height'] < 16:
                    continue
 
               

                info['path_to_image'] = path_to_image
                info['id'] = i
                
                info['question'] = "[obj] Describe this object."
                info['answer'] = entry['caption'][int(chunk[0]):int(chunk[1])]
                
                self.entries.append(info)
        return self.entries

    def _get_ViT_mask(self, bbox, height, width):
        arr = np.zeros((self.patch_size, self.patch_size))
        x_min, y_min, x_max, y_max = bbox
        height_bins = np.linspace(0, height, self.patch_size)
        width_bins = np.linspace(0, width, self.patch_size)
        x_min, x_max = np.digitize(np.array([x_min, x_max]), width_bins)
        y_min, y_max = np.digitize(np.array([y_min, y_max]), height_bins)

        if y_min == y_max:
            y_max += 1
        if x_min == x_max:
            x_max += 1
        arr[y_min:y_max, x_min:x_max] = 1



        return np.append(1, arr.flatten())
    
    def collate_fn(self, batch):
        
        
        return {
            "id": [item["id"] for item in batch],
            'path_to_image': [item["path_to_image"] for item in batch],
            'bbox': [item["bbox"] for item in batch],
            'question': [item["question"] for item in batch],
            'vit_mask': [item["vit_mask"] for item in batch],
            "answer":  [item["answer"] for item in batch],
        }

    def __str__(self):
        return f"GRIT pretraining dataset with {len(self.entries)} questions"


    def __len__(self):
        return len(self.entries)


    def __getitem__(self, index):
        entry = self.entries[index]


        x_min, y_min, x_max, y_max = entry['chunk'][2:6]
        x_min = x_min * entry['width']
        x_max = x_max * entry['width']
        y_min = y_min * entry['height']
        y_max = y_max * entry['height']
        entry['vit_mask'] = self._get_ViT_mask([x_min, y_min, x_max, y_max], entry['height'], entry['width'])
        entry['bbox'] = [x_min, y_min, x_max - x_min, y_max-y_min]

        return entry