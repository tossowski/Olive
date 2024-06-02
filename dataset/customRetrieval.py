import json
import os
import numpy as np
import pycocotools.mask as mask
from PIL import Image
import math
import torch
import skimage
import pickle
import random
from tqdm import tqdm
from torch.utils.data import Dataset


# A class to allow you to create a custom retrieval dataset
# Make sure you have a folder with images in it. 
# Each image should be named x.png or x.jpg, where x is a 
# whole number.
# The folder should also have a file called labels.txt in it,
# where line number x contains the label for x.png or x.jpg
# See additional_examples folder for an example of this
#
# Once you have created your folder, run python retrieve.py --train
# with additional_retrieval_examples filled in the config to create
# the retrieval set
class RetrievalDataset(Dataset):
    def __init__(self, config):
        super(RetrievalDataset, self).__init__()
        self.config = config
        self.class_counts = {}
        self.n_patches = 16
        if "336" in config['vision_encoder']:
            self.n_patches = 24
        self.entries = self._load_dataset()
        

        
    
    def _load_dataset(self):
        entries = []
        labels = []
        path_to_images = self.config["additional_retrieval_examples"]
        with open(os.path.join(path_to_images, "labels.txt"), "r") as f:
            for line in f:
                labels.append(line.strip())

        image_filenames = [x for x in os.listdir(path_to_images) if x.endswith(".png") or x.endswith(".jpg")]
        all_images = sorted(image_filenames, key = lambda x: int(x.split(".")[0]))
        i = 0
        for f_name in tqdm(all_images):
            path_to_image = os.path.join(path_to_images, f_name)
            if not (path_to_image.endswith("png") or path_to_image.endswith("jpg")):
                continue
            image = Image.open(path_to_image)
            segmentations = np.ones(self.n_patches ** 2 + 1)
            bboxes = [0, 0, image.width, image.height]
            segmentation_labels = labels[i]
                
            if labels[i] not in self.class_counts:
                self.class_counts[labels[i]] = 0
            self.class_counts[labels[i]] += 1
            entries.append({"id": None, "path_to_image": path_to_image, "question": "[obj] What is this?", "vit_mask": segmentations, "answer": segmentation_labels, "original_segmentation": None, 'bbox':bboxes})
            i += 1
        return entries

    def collate_fn(self, batch):
        return {
            "id": [item["id"] for item in batch],
            'path_to_image': [item["path_to_image"] for item in batch],
            'bbox': [item["bbox"] for item in batch],
            'question': [item["question"] for item in batch],
            'vit_mask': [item["vit_mask"] for item in batch],
            "answer":  [item["answer"] for item in batch],
            "original_segmentation":  [item["original_segmentation"] for item in batch]
        }
        
    def stats(self):
        class_counts = {}
        for example in self.entries:
            c = example['answer']

            if c not in class_counts:
                class_counts[c] = 0
            class_counts[c] += 1
        
        for key in class_counts:
            class_counts[key] = round(class_counts[key] / len(self.entries), 2)
        return class_counts



    def __str__(self):
        return f"Retrieval dataset with {len(self.entries)} entries"


    def __len__(self):
        return len(self.entries)


    def __getitem__(self, index):
        entry = self.entries[index]
        return entry