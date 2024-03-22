import json
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

class OCRDataset(Dataset):
    # Path to instances.json
    # Path to COCO2014/COCO2017 train images
    def __init__(self, config, split="train", patch_size=16):
        super(OCRDataset, self).__init__()
        self.split = split
        self.data = json.load(open("/data/ossowski/OCR/TextVQA_Rosetta_OCR_v0.2_train.json"))['data']
        self.config = config
        self.patch_size = patch_size
        self.entries = self._load_dataset()

    def _get_ViT_mask(self, bbox, height, width):
        arr = np.zeros((self.patch_size, self.patch_size))
        x_min, y_min, w, h = bbox
        y_max = y_min + h
        x_max = x_min + w
        height_bins = np.linspace(0, height, self.patch_size)
        width_bins = np.linspace(0, width, self.patch_size)
        #print(x_min, x_max, y_min, y_max)
        x_min, x_max = np.digitize(np.array([x_min, x_max]), width_bins)
        y_min, y_max = np.digitize(np.array([y_min, y_max]), height_bins)
        #print(x_min, x_max, y_min, y_max)

        # x_max += 1
        # y_max += 1
        if y_min == y_max:
            y_max += 1
        if x_min == x_max:
            x_max += 1
        arr[y_min:y_max, x_min:x_max] = 1

        return np.append(1, arr.flatten())
    
    def _load_dataset(self):

        entries = []
        bad_segs = 0
        for entry in tqdm(self.data):
            chunk = []
            id = entry['image_id']
            for q in entry['ocr_info']:

                path = os.path.join("/data/ossowski/OCR/train_images", id + ".jpg")
                image = Image.open(path)
                x = q['bounding_box']['top_left_x'] * image.width
                y = q['bounding_box']['top_left_y'] * image.height
                w = q['bounding_box']['width'] * image.width
                h = q['bounding_box']['height'] * image.height
                bbox = [x, y, w, h]
                mask = self._get_ViT_mask(bbox, image.height, image.width)

                question = "[obj] What text is written here?"
                label = q['word']

                
                

                chunk.append({"id": id, "path_to_image": path, "question": question, "vit_mask": mask, "answer": label,  'bbox':bbox})
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
        }

    
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

    
    def __str__(self):
        return f"PointQA dataset with {len(self.entries)} questions"


    def __len__(self):
        return len(self.entries)


    def __getitem__(self, index):
        entry = self.entries[index]

        return entry