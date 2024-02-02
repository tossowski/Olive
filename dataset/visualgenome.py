import json
import os
import numpy as np
import pycocotools.mask as mask
import math
import torch
import skimage
import pickle
import datasets

from tqdm import tqdm
from torch.utils.data import Dataset


class VisualGenomeDataset(Dataset):
    # Path to instances.json
    # Path to COCO2014/COCO2017 train images
    def __init__(self, config, split="train", patch_size=16, max_examples_per_class = 1000):
        super(VisualGenomeDataset, self).__init__()
        self.config = config
        datasets.config.DOWNLOADED_DATASETS_PATH = "/data/ossowski/downloads"
        self.dataset = datasets.load_dataset("visual_genome", "relationships_v1.2.0")
        self.patch_size = patch_size
        self.max_examples_per_class = max_examples_per_class
        self.class_counts = {}
        self.entries = self._load_dataset()

        if config["use_retrieval"]:
            raise NotImplementedError
    
    def _get_ViT_mask(self, bbox, height, width):
        arr = np.zeros((self.patch_size, self.patch_size))
        x_min, y_min, x_max, y_max = bbox
        height_bins = np.linspace(0, height, 16)
        width_bins = np.linspace(0, width, 16)
        x_min, x_max = np.digitize(np.array([x_min, x_max]), width_bins)
        y_min, y_max = np.digitize(np.array([y_min, y_max]), height_bins)
        arr[y_min:y_max + 1, x_min:x_max] = 1

        return np.append(1, arr.flatten())
    
    def _load_dataset(self):

        entries = []
        bad_pairs = 0
        for item in tqdm(self.dataset['train']):
            width = item['width']
            height = item['height']
            chunk = []
            for rel in item['relationships']:
                subject = rel['subject']
                object = rel['object']
                predicate = rel['predicate']
                s_bbox = (subject['x'], subject['y'], subject['x'] + subject['w'], subject['y'] + subject['h'])
                o_bbox = (object['x'], object['y'], object['x'] + object['w'], object['y'] + object['h'])

                seg_o = self._get_ViT_mask(o_bbox, height, width)
                seg_s = self._get_ViT_mask(s_bbox, height, width)
                if sum(seg_o[1:]) == 0 or sum(seg_s[1:]) == 0:
                    bad_pairs += 1
                    continue

                label = f"{subject['names'][0]} {predicate} {object['names'][0]}"
                chunk.append({"path_to_image": [item["image"], item["image"]], "question": "[obj] [obj] What is the relationship between these objects?", "vit_mask": [seg_o, seg_s], "answer": label})
            
            entries.extend(chunk)

        print(f"Skipped over {bad_pairs} bad pairs (no pixels)")
        return entries

    def collate_fn(self, batch):
        return {
            'path_to_image': [item["path_to_image"] for item in batch],
            'question': [item["question"] for item in batch],
            'vit_mask': [item["vit_mask"] for item in batch],
            "answer":  [item["answer"] for item in batch],
        }

    def eval_correctness(self, prediction, answer):
        correct = 1 if prediction == answer else 0
        score_dict = {}
        score_dict["score"] = correct
        return  score_dict

    def stats(self):
        return "Not Implemented"

    # Should take in argument k: how many closest objects to retrieve
    # and object features: the ViT features of query objects
    # Return: The k closest examples from self.entries
    def retrieve_closest(self, object_features, k, train_phase = True):

        return None
    


    def __str__(self):
        return f"Visual Genome dataset with {len(self.entries)} questions"


    def __len__(self):
        return len(self.entries)


    def __getitem__(self, index):
        entry = self.entries[index]

        return entry