import json
from torch.utils.data import Dataset

import os
from matplotlib import pyplot as plt
import numpy as np

class COCOCaptionsDataset(Dataset):
    def __init__(self, image_path, caption_path, labels_path, split = "train"):
        super(COCOCaptionsDataset, self).__init__()
        self.image_path = image_path
        self.info = json.load(open("/data/ossowski/COCO2017/info.json"))
        self.image_id_dict = {}
        self.split = split
        self.image_ids = self._load_dataset(caption_path, labels_path)
        

    def _load_dataset(self, captions_path, annotations_path):
        #print("here")
        annotations = json.load(open(annotations_path))
        captions = json.load(open(captions_path))

        entries = []
        
        for image in annotations['images']:
            image['objects'] = []
            image['crowds'] = []
            image['captions'] = []
            image['segmentations'] = []
            image['segmentation_labels'] = []
            image['bboxes'] = []

            self.image_id_dict[image['id']] = image

        # Adding segmented objects to data
        for annotation in annotations["annotations"]:
            
            d = {}
            image_id = annotation['image_id']
            self.image_id_dict[image_id]['bboxes'].append(annotation['bbox'])
            self.image_id_dict[image_id]['segmentations'].append(annotation['segmentation'])
            self.image_id_dict[image_id]['segmentation_labels'].append(self.info['classes'][annotation['category_id']])
            
                
            # elif type(annotation['segmentation']) == dict:
            #     rle = annotation['segmentation']
            #     compressed_rle = mask.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
            #     m = mask.decode(compressed_rle)
            #     d['label'] = 'crowd'
            #     d['mask'] = m
            #     self.image_id_dict[image_id]['crowds'].append(d)
            # else:
            #     print("no annotation")
            #print(mask.decode({size: [], 'counts': annotation['segmentation'][0]}))

        for caption in captions['annotations']:
            self.image_id_dict[caption['image_id']]['captions'].append(caption['caption'])
            
        entries = sorted(self.image_id_dict.keys())

      
        return entries


    def __str__(self):
        return f"COCO Captioning Dataset with {len(self.image_ids)} images"


    def __len__(self):
        return len(self.image_ids)


    def __getitem__(self, index):
        data = self.image_id_dict[self.image_ids[index]]
        item = {}
        item['file_name'] = data['file_name']
        item['image_id'] = self.image_ids[index]

        return item