import json
import os
import numpy as np
import pycocotools.mask as mask
import math
import torch
import skimage
import pickle
import random
from tqdm import tqdm
from torch.utils.data import Dataset


class COCOObjectDataset(Dataset):
    # Path to instances.json
    # Path to COCO2014/COCO2017 train images
    def __init__(self, config, split="train", patch_size=16, max_examples_per_class = 1000000):
        super(COCOObjectDataset, self).__init__()
        dataset_path = f"/data/ossowski/COCO2017/instruction_data/supervised_with_segmentations_{split}_object_detection_{patch_size}x{patch_size}.json"
        self.config = config
        self.data = json.load(open(dataset_path))
        self.prompts = ["[obj] What is this? Answer as short as possible.",
                        "[obj] What is this object? Answer as short as possible."]
        self.patch_size = patch_size
        self.max_examples_per_class = max_examples_per_class
        self.class_counts = {}
        self.entries = self._load_dataset()

        if config["use_retrieval"]:
            if config["crop_image"]:
                cropped = "cropped_"
            else:
                cropped = ""
            retrieval_path = f'retrieval/{config["task"]}/retrieval_set_{config["examples_per_class"]}_{cropped}{config["vision_encoder"].split("/")[-1]}.pkl'
            #retrieval_path = f'retrieval/{config["task"]}/retrieval_set_{config["examples_per_class"]}_{config["vision_encoder"].split("/")[-1]}.pkl'

            #retrieval_path = "retrieval/object_classification/retrieval_set_1000000_1_clip-vit-large-patch14.pkl"
            #retrieval_path = "retrieval/object_classification/retrieval_set_1000000_dinov2-large.pkl"
            with open(retrieval_path, 'rb') as f:
                self.retrieval_data = pickle.load(f)
                self.retrieval_keys = torch.FloatTensor(self.retrieval_data['keys']).to(self.config["device"])
                self.retrieval_labels = self.retrieval_data['values']
                self.retrieval_idx = self.retrieval_data['idx']
                assert len(self.retrieval_keys) == len(self.retrieval_labels)
                print(f'Loaded {len(self.retrieval_keys)} examples from {retrieval_path}.pkl')
    
    def _load_dataset(self):

        entries = []
        bad_segs = 0
        for item in tqdm(self.data):
            segmentations = item["segmentations"]
            bboxes = item["bboxes"]
            segmentation_labels = item["segmentation_labels"]
            full_segmentations = item["original_segmentations"]

            chunk = []
            
            for i, segmentation in enumerate(segmentations):
                bbox = bboxes[i]
                #insert = 1 if self.config["use_CLS_emb"] else 0
                insert = 1
                seg = np.append(insert, mask.decode(segmentation).flatten())
                if sum(seg[1:]) == 0:
                    bad_segs += 1
                    continue
                original_seg = full_segmentations[i]
                label = segmentation_labels[i]
                if label.isnumeric():
                    continue
                if label not in self.class_counts:
                    self.class_counts[label] = 0
                if self.class_counts[label] == self.max_examples_per_class:
                    continue
                self.class_counts[label] += 1
                chunk.append({"id": item["id"], "path_to_image": item["image"], "question": "[obj] What is this?", "vit_mask": seg, "answer": label, "original_segmentation": original_seg, 'bbox':bbox})
            entries.extend(chunk)
        print(f"Skipped over {bad_segs} bad segmentations (no pixels)")
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

    def eval_correctness(self, prediction, answer):
        correct = 1 if prediction == answer else 0
        score_dict = {}
        score_dict["score"] = correct
        return  score_dict
    
    def get_entry_by_id(self, id):
        for entry in self.entries:
            if entry['id'] == id:
                return entry
        
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

    # Should take in argument k: how many closest objects to retrieve
    # and object features: the ViT features of query objects
    # Return: The k closest examples from self.entries
    def retrieve_closest(self, object_features, k, train_phase = True, b_num=-1):

        dist_matrix = (object_features @ self.retrieval_keys.T)
        #print(dist_matrix.shape)
        #print(len(torch.nonzero(torch.isnan(self.retrieval_keys.view(-1)))))


        if train_phase:
            closest_indices = torch.argsort(dist_matrix, axis = -1, descending=True)[:, 1:1+k]
        else:
            closest_indices = torch.argsort(dist_matrix, axis = -1, descending=True)[:, 0:k]
        #print(closest_indices)
        similarity_scores = [[round(dist_matrix[i, x].item(), 2) for x in closest_indices[i,:]] for i in range(len(closest_indices))]
        #print([[dist_matrix[i, x] for x in closest_indices[i,:]] for i in range(len(closest_indices))])
        #print(similarity_scores)
        retrieved_info = [[self.get_entry_by_id(self.retrieval_idx[x]) for x in closest_indices[i,:]] for i in range(len(closest_indices))]

        print(similarity_scores)

        return retrieved_info, similarity_scores


    def __str__(self):
        return f"ObjectCOCO dataset with {len(self.entries)} questions"


    def __len__(self):
        return len(self.entries)


    def __getitem__(self, index):
        entry = self.entries[index]

        return entry