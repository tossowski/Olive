import json
import os
import numpy as np
import pycocotools.mask as mask
import math
import torch
import skimage
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

class ObjectInstructDataset(Dataset):
    # Path to instances.json
    # Path to COCO2014/COCO2017 train images
    def __init__(self, config, split="train", n_patches=16, max_examples_per_class = 1000000):
        super(ObjectInstructDataset, self).__init__()
        self.split = split
        dataset_path = "/work/ossowski/alignment/dataset/vip-llava_stage3_mix.json"

        self.config = config
        self.data = json.load(open(dataset_path))

        self.n_patches = n_patches
        self.class_counts = {}
        self.entries = self._load_dataset()

    def _get_ViT_mask_from_segmentation(self, segmentation, height, width):
        
        output_width, output_height = self.n_patches, self.n_patches
        rles = mask.frPyObjects(segmentation, height, width)
        rle = mask.merge(rles)
        m =  mask.decode(rle).astype(bool)

        rle['counts'] = rle['counts'].decode('ascii')
        pooled_mask = skimage.measure.block_reduce(m, block_size=(math.floor(height / self.n_patches), math.floor(width / self.n_patches)), func=np.max)


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
    
    def _get_ViT_mask_from_bbox(self, bbox, height, width):
        arr = np.zeros((self.n_patches, self.n_patches))
        x_min, y_min, x_max, y_max = bbox
        height_bins = np.linspace(0, height, self.n_patches)
        width_bins = np.linspace(0, width, self.n_patches)
        x_min, x_max = np.digitize(np.array([x_min, x_max]), width_bins)
        y_min, y_max = np.digitize(np.array([y_min, y_max]), height_bins)

        if y_min == y_max:
            y_max += 1
        if x_min == x_max:
            x_max += 1
        arr[y_min:y_max, x_min:x_max] = 1

        return np.append(1, arr.flatten())
    
    def _load_dataset(self):

        entries = []
        bad_segs = 0
        import re
        for item in tqdm(self.data):
            if 'gpt4v' not in str(item['id']):
                continue
            
            if 'gpt4v' in str(item['id']):
                if "coco" in item["image"]:
                    filename = item["image"].split("/")[-1]
                    if "train" in item["image"]:
                        path_to_image = os.path.join("/data/ossowski/COCO2017/train/data", filename)
                    elif "val" in item["image"]:
                        path_to_image = os.path.join("/data/ossowski/COCO2017/val/data", filename)
                elif "VG_100K" in item["image"]:
                    path_to_image = os.path.join("/data/ossowski", item['image'])
                
                image = Image.open(path_to_image)
                height = image.height
                width = image.width
                # if 'segmentations' in item:
                #     segs = item['segmentations']
                #     vit_masks = [np.append(1, self._get_ViT_mask_from_segmentation(seg, height, width).flatten()) for seg in segs]
                #     vit_masks = np.stack(vit_masks, axis = 0)
                # else:
                bboxes = item['bboxes']
                masks = [self._get_ViT_mask_from_bbox(bbox, height, width) for bbox in bboxes]
                for i in range(len(item['conversations']) // 2):
                    q_info = item['conversations'][i * 2]
                    a_info = item['conversations'][i * 2 + 1]
                    assert q_info['from'] == 'human'
                    assert a_info['from'] == 'gpt'
                    raw_question = q_info['value']
                    pattern = r'<bbox(?:\d*)>'

                    # Find all matches of the pattern in the text
                    matches = re.findall(pattern, raw_question)
                    # Convert the matched strings to integers
                    # print(raw_question)
                    # print(matches)
                    numbers = [int(match[5:-1]) if match != "<bbox>" else 0 for match in matches]
                    question_masks = [masks[x] for x in numbers]
                    if len(question_masks) == 0:
                        question_masks = [np.ones(self.n_patches * self.n_patches + 1)]
                        raw_question = "[obj] " + raw_question
                    vit_masks = np.stack(question_masks, axis = 0)
                    final_question = re.sub(pattern, '[obj]', raw_question)
                    ##print(final_question)


                    raw_answer = a_info['value']
                    pattern = r' <bbox\d+>'
                    final_answer = re.sub(pattern, '', raw_answer)

                    pattern = r' <bbox>'
                    final_answer = re.sub(pattern, '', final_answer)
                    #final_answer = final_answer.replace("[obj]", "")

                    entries.append({"id": item["id"], "path_to_image": [path_to_image] * len(vit_masks), "question": final_question, "vit_mask": vit_masks, "answer": final_answer, 'bbox':item['bboxes']})

        
        return entries

    def collate_fn(self, batch):
        return {
            "id": [item["id"] for item in batch],
            'path_to_image': [item["path_to_image"] for item in batch],
            'bbox': [item["bbox"] for item in batch],
            'question': [item["question"] for item in batch],
            'vit_mask': [item["vit_mask"] for item in batch],
            "answer":  [item["answer"] for item in batch],
        }

        
    def stats(self):
        class_counts = {}
        for example in self.entries:
            c = example['answer']

            if c not in class_counts:
                class_counts[c] = 0
            class_counts[c] += 1
        
        #for key in class_counts:
            #class_counts[key] = round(class_counts[key] / len(self.entries), 2)
            # class_counts[key] = round(class_counts[key] / len(self.entries), 2)


        return class_counts

   

    def __str__(self):
        return f"Object Instruction Tuning dataset with {len(self.entries)} questions"


    def __len__(self):
        return len(self.entries)


    def __getitem__(self, index):
        entry = self.entries[index]

        return entry