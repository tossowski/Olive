import json
import os
import numpy as np
import pycocotools.mask as mask
import math
import skimage
import pickle
import random
from tqdm import tqdm
from torch.utils.data import Dataset


class VRPDataset(Dataset):
    # Path to instances.json
    # Path to COCO2014/COCO2017 train images
    def __init__(self, split="train", patch_size=16, use_object_annotations = True):
        super(VRPDataset, self).__init__()
        self.patch_size = patch_size
        self.use_object_annotations = use_object_annotations
        self.object_prompts = ["Describe the image.",
                                "Explain the image with these objects.",
                                "Using these objects, describe the image.",
                                "Describe the image using these things.",
                                "Tell me about the image with the help of these objects.",
                                "Use these things to paint a picture of what you see.",
                                "Break down the image with these specified objects.",
                                "Use these things to describe the image.",
                                "Spell out the image using these specific items.",
                                "Share your thoughts on the picture using these things."]
        if split in ["train", "val"]:
            dataset_path = f"/data/ossowski/COCO2017/instruction_data/supervised_with_segmentations_{split}_object_detection_{patch_size}x{patch_size}.json"
            self.data = json.load(open(dataset_path))
            

            caption_dataset_path = f"/data/ossowski/COCO2017/annotations/captions_{split}2017.json"
            
            self.caption_data = json.load(open(caption_dataset_path))

            self.image_id_to_caption = {}
            for annotation in self.caption_data['annotations']:
                self.image_id_to_caption[annotation['image_id']] = annotation['caption']
        else:
            
            caption_dataset_path = f"/data/ossowski/COCO2017/annotations/image_info_{split}2014.json"
            self.data = json.load(open(caption_dataset_path))

        self.entries = self._load_dataset(split)
    
    def _load_dataset(self, split):

        entries = []
        invalid = 0
        if split in ["train", "val"]:
            
            for item in self.data:
                bboxes = item["bboxes"]
                segmentations = item["segmentations"]
                
                if len(segmentations) <= 1:
                    continue

                segmentation_labels = item["segmentation_labels"]

                image_id = int(item['id'])
                caption = self.image_id_to_caption[image_id]

                # Old code assuming ground truth masks
                if self.use_object_annotations:
                    vit_masks = [np.append(1, mask.decode(seg).flatten()) for seg in segmentations]
                    vit_masks = np.stack(vit_masks, axis = 0)
                    question = "[obj] " * len(segmentation_labels) + random.choice(self.object_prompts)

                else:

                    vit_masks = np.ones(self.patch_size * self.patch_size + 1)
                    question = "[obj] Describe the image."

                # if len(segmentations) != len(segmentation_labels):
                #     print(segmentation_labels)
                #     print(image_id)
                #     invalid += 1
                #     continue
                # assert len(vit_masks) == len(segmentation_labels)
                
                # Old code assuming ground truth masks
                #entry = {"path_to_image": item["image"], "question": "[obj] " * len(segmentation_labels) + "Describe the image using these objects.", "vit_mask": vit_masks, "answer": caption}
                
                entry = {"id": image_id, "path_to_image": item["image"], "question": question, "vit_mask": vit_masks, "answer": caption, "bbox": bboxes}

                entries.append(entry)
        else:
            for item in self.data['images']:
                image_id = int(item['id'])
                path_to_image = os.path.join("/data/ossowski/COCO2017/test2014", item['file_name'])
                vit_masks = np.ones((1, self.patch_size * self.patch_size + 1))
                entry = {"id": image_id, "path_to_image": path_to_image, "question": "[obj] Describe the image in 1 sentence.", "vit_mask": vit_masks, "answer": "Answer not given in test data"}

                entries.append(entry)
        print(f"{invalid} Invalid entries")
        return entries

    def collate_fn(self, batch):
        return {
            'path_to_image': [item["path_to_image"] for item in batch],
            'question': [item["question"] for item in batch],
            'vit_mask': [item["vit_mask"] for item in batch],
            "answer":  [item["answer"] for item in batch],
            "id":  [item["id"] for item in batch]
        }

    def stats(self):
        
        return "Not Implemented Yet"

    def __str__(self):
        return f"VRP dataset with {len(self.entries)} questions"


    def __len__(self):
        return len(self.entries)


    def __getitem__(self, index):
        entry = self.entries[index]

        return entry