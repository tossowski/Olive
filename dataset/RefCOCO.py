import json
import os
import numpy as np
import pycocotools.mask as mask
import math
import skimage
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset
import random

class RefCOCODataset(Dataset):
    # Path to instances.json
    # Path to COCO2014/COCO2017 train images
    def __init__(self, data_root, train_image_path, patch_size=16, split='train'):
        super(RefCOCODataset, self).__init__()
        self.data = json.load(open(os.path.join(data_root, "instances.json") , "r"))
        self.patch_size = patch_size
        self.ref_data = pickle.load(open(os.path.join(data_root, "refs(google).p"), "rb"))
        self.image_path = train_image_path
        self.prompts = ["[obj] Briefly describe this image region",
                        "[obj] Describe this part of the image.",
                        "[obj] Share some details about what's happening here in the image.",
                        "[obj] Break down what you see in this particular part of the picture.",
                        "[obj] Fill me in on the details of this section in the picture.",
                        "[obj] Describe what you notice in this area of the picture."]
        self.split = split
        self.image_dict = {}
        for image in self.data['images']:
            self.image_dict[image['id']] = image
        self.ann_dict = {}
        for ann in self.data['annotations']:
            self.ann_dict[ann['id']] = ann
        self.label_dict = {}
        for cat in self.data['categories']:
            self.label_dict[cat['id']] = cat['name']
        self.entries = self._load_dataset()

    def _get_ViT_mask(self, segmentation, height, width):
        
        output_width, output_height = self.patch_size, self.patch_size
        rles = mask.frPyObjects(segmentation, height, width)
        rle = mask.merge(rles)
        m =  mask.decode(rle).astype(bool)

        rle['counts'] = rle['counts'].decode('ascii')
        pooled_mask = skimage.measure.block_reduce(m, block_size=(math.floor(height / self.patch_size), math.floor(width / self.patch_size)), func=np.max)


        result_height, result_width = pooled_mask.shape
        # If the result is smaller than 16x16, pad it with zeros
        if result_height < output_height or result_width < output_width:
            pad_height = output_height - result_height
            pad_width = output_width - result_width
            pooled_mask = np.pad(pooled_mask, ((0, pad_height), (0, pad_width)), mode='constant')

        if result_height > output_height or result_width > output_width:
            pooled_mask = pooled_mask[:output_height, :output_width]

        assert pooled_mask.shape == (output_height,output_width)
        return pooled_mask

    def _load_dataset(self):

        entries = []
        for entry in tqdm(self.ref_data):
            
            if entry['split'] != self.split:
                continue
            #print(entry['split'])
            # for sent in entry['sentences']:
            sample_info = {}
            image_id = entry['image_id']
            path_to_image = f"/data/ossowski/COCO2017/train/data/{self.image_dict[image_id]['file_name'].split('_')[-1]}"
            annotation = self.ann_dict[entry['ann_id']]
            segmentation = annotation['segmentation']
            category_id = self.ann_dict[entry['ann_id']]['category_id']
            label = self.label_dict[category_id]
            image_info = self.image_dict[annotation['image_id']]
            height = image_info['height']
            width = image_info['width']

            vit_mask = self._get_ViT_mask(segmentation, height, width)
            sample_info['original_segmentation'] = segmentation
            sample_info['vit_mask'] = np.append(1, vit_mask.flatten())
            sample_info['id'] = image_id
            sample_info['label'] = label
            sample_info['sentence'] = entry['sentences'][0]['raw']
            sample_info['refs'] = [entry['sentences'][x]['raw'] for x in range(len(entry['sentences']))]
            sample_info['bbox'] = annotation['bbox']
            sample_info['image_height'] = height
            sample_info['image_width'] = width
            sample_info['path_to_image'] = path_to_image
            
            sample_info['question'] = random.choice(self.prompts)
            sample_info['answer'] = sample_info['sentence']
            entries.append(sample_info)
        return entries

    def collate_fn(self, batch):
        return {
            "id": [item["id"] for item in batch],
            'path_to_image': [item["path_to_image"] for item in batch],
            'question': [item["question"] for item in batch],
            'vit_mask': [item["vit_mask"] for item in batch],
            'bbox': [item["bbox"] for item in batch],
            "answer":  [item["answer"] for item in batch],
            "refs": [item["refs"] for item in batch],
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
        return f"RefCOCO {self.split} dataset with {len(self.entries)} questions"


    def __len__(self):
        return len(self.entries)


    def __getitem__(self, index):
        entry = self.entries[index]

        return entry