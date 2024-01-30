import json
import os
import numpy as np
import pycocotools.mask as mask
import math
import random
import skimage
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset


class CountCOCODataset(Dataset):
    # Path to instances.json
    # Path to COCO2014/COCO2017 train images
    def __init__(self, split="train", patch_size=16):
        super(CountCOCODataset, self).__init__()
        random.seed(42)
        dataset_path = f"/data/ossowski/COCO2017/instruction_data/supervised_with_segmentations_{split}_object_detection_{patch_size}x{patch_size}.json"
        self.data = json.load(open(dataset_path))
        self.info = json.load(open("/data/ossowski/COCO2017/info.json"))

        caption_dataset_path = f"/data/ossowski/COCO2017/annotations/captions_{split}2017.json"
        self.caption_data = json.load(open(caption_dataset_path))

        self.image_id_to_caption = {}
        for annotation in self.caption_data['annotations']:
            self.image_id_to_caption[annotation['image_id']] = annotation['caption']

        self.entries = self._load_dataset()
    
    def _load_dataset(self):

        entries = []
        invalid = 0
        for item in tqdm(self.data):
            segmentations = item["segmentations"]
            
            if len(segmentations) <= 1:
                continue

            segmentation_labels = item["segmentation_labels"]

            image_id = int(item['id'])
            caption = self.image_id_to_caption[image_id]

            vit_masks = [np.append(mask.decode(seg).flatten(), 1) for seg in segmentations]
            
            vit_masks = np.stack(vit_masks, axis = 0)

            if len(segmentations) != len(segmentation_labels):
                print(segmentation_labels)
                print(image_id)
                invalid += 1
                continue
            assert len(vit_masks) == len(segmentation_labels)

            if len(segmentation_labels) > 40:
                continue
            objs_in_image = list(set(segmentation_labels))
            valid_values = [value for value in self.info['classes'] if value not in objs_in_image]
            #original_segmentations = [mask.decode(x) for x in item['original_segmentations']]
            original_segmentations = item['original_segmentations']


            #print(original_segmentations[0])
            sampled_values = random.sample(valid_values, len(objs_in_image))
            for i, obj in enumerate(objs_in_image):
                
                obj_indices = [str(i) for i, x in enumerate(segmentation_labels) if x == obj]
                original_segs = [original_segmentations[i] for i, x in enumerate(segmentation_labels)]
                if len(obj_indices) > 1:
                    answer = f"There are {len(obj_indices)} {obj}s at " + " ".join(obj_indices)
                else:
                    answer = f"There is {len(obj_indices)} {obj} at " + " ".join(obj_indices)
                entry = {"path_to_image": item["image"], "question": "[obj] " * len(segmentation_labels) + f"How many {obj}s are in this image?", "vit_mask": vit_masks, "answer": answer, "original_segmentations": original_segs}

                entries.append(entry)

                negative_entry = {"path_to_image": item["image"], "question": "[obj] " * len(segmentation_labels) + f"How many {sampled_values[i]}s are in this image?", "vit_mask": vit_masks, "answer": f"There is no {sampled_values[i]} in this image", "original_segmentations": []}
                entries.append(negative_entry)
        print(f"{invalid} Invalid entries")
        return entries

    def eval_correctness(self, prediction, answer):

        numbers = [int(x) for x in prediction.split(" ") if x.isdigit()]
        answer_numbers = [int(x) for x in answer.split(" ") if x.isdigit()]
        info_dict = {"score": 0, "num_object_correct": 0}
    
        if prediction.startswith("There is no"):
            if len(answer_numbers) == 0:
                info_dict['score'] = 1
                info_dict["num_object_correct"] = 1
            else:
                info_dict['score'] = 0
                info_dict["num_object_correct"] = 0
            return info_dict
            
        try:
            if numbers[0] == answer_numbers[0]:
                info_dict["num_object_correct"] = 1


            total = 0
            for num in numbers[1:]:
                if num in answer_numbers[1:]:
                    total += 1
            
            info_dict['score'] = total / (len(numbers) - 1)
            return info_dict
        except:
            info_dict['score'] = 0
            info_dict["num_object_correct"] = 0
            return info_dict


    def collate_fn(self, batch):
        return {
            'path_to_image': [item["path_to_image"] for item in batch],
            'question': [item["question"] for item in batch],
            'vit_mask': [item["vit_mask"] for item in batch],
            "answer":  [item["answer"] for item in batch]
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