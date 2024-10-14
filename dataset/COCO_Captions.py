import json
from torch.utils.data import Dataset

import os
from matplotlib import pyplot as plt
import numpy as np

class COCOCaptionsDataset(Dataset):
    def __init__(self, image_path, caption_path, labels_path, split = "train"):
        super(COCOCaptionsDataset, self).__init__()
        self.image_path = image_path
        self.info = {
            "name": "coco-2017",
            "zoo_dataset": "fiftyone.zoo.datasets.base.COCO2017Dataset",
            "dataset_type": "fiftyone.types.dataset_types.COCODetectionDataset",
            "num_samples": 123287,
            "downloaded_splits": {
                "train": {
                    "split": "train",
                    "num_samples": 118287
                },
                "validation": {
                    "split": "validation",
                    "num_samples": 5000
                }
            },
            "classes": [
                "0",
                "person",
                "bicycle",
                "car",
                "motorcycle",
                "airplane",
                "bus",
                "train",
                "truck",
                "boat",
                "traffic light",
                "fire hydrant",
                "12",
                "stop sign",
                "parking meter",
                "bench",
                "bird",
                "cat",
                "dog",
                "horse",
                "sheep",
                "cow",
                "elephant",
                "bear",
                "zebra",
                "giraffe",
                "26",
                "backpack",
                "umbrella",
                "29",
                "30",
                "handbag",
                "tie",
                "suitcase",
                "frisbee",
                "skis",
                "snowboard",
                "sports ball",
                "kite",
                "baseball bat",
                "baseball glove",
                "skateboard",
                "surfboard",
                "tennis racket",
                "bottle",
                "45",
                "wine glass",
                "cup",
                "fork",
                "knife",
                "spoon",
                "bowl",
                "banana",
                "apple",
                "sandwich",
                "orange",
                "broccoli",
                "carrot",
                "hot dog",
                "pizza",
                "donut",
                "cake",
                "chair",
                "couch",
                "potted plant",
                "bed",
                "66",
                "dining table",
                "68",
                "69",
                "toilet",
                "71",
                "tv",
                "laptop",
                "mouse",
                "remote",
                "keyboard",
                "cell phone",
                "microwave",
                "oven",
                "toaster",
                "sink",
                "refrigerator",
                "83",
                "book",
                "clock",
                "vase",
                "scissors",
                "teddy bear",
                "hair drier",
                "toothbrush"
            ]
        }
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