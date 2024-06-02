import os
import numpy as np
import torch
import pickle
import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

# Medical Chest X-Ray Dataset
class CXR8Dataset(Dataset):
    def __init__(self, config, split="train", n_patches=16):
        super(CXR8Dataset, self).__init__()
        self.config = config

        self.split=split
        self.n_patches = n_patches
        
        self.class_counts = {}
        self.entries = self._load_dataset()

        if config["use_retrieval"]:
            if config["crop_image"]:
                cropped = "cropped_"
            else:
                cropped = ""
            retrieval_path = f'retrieval/{config["task"]}/retrieval_set_{config["examples_per_class"]}_{cropped}{config["vision_encoder"].split("/")[-1]}.pkl'
            if "retrieval_set_path" in config:
                retrieval_path = config["retrieval_set_path"]


            if os.path.exists(retrieval_path):
                with open(retrieval_path, 'rb') as f:
                    
                    self.retrieval_data = pickle.load(f)
                    self.retrieval_keys = torch.FloatTensor(self.retrieval_data['keys']).to(self.config["device"])
                    self.retrieval_labels = self.retrieval_data['values']
                    self.retrieval_idx = self.retrieval_data['idx']
                    assert len(self.retrieval_keys) == len(self.retrieval_labels)
                    print(f'Loaded {len(self.retrieval_keys)} examples from {retrieval_path}.pkl')
    
    # Given a bounding box, get the Vision Transformer binary mask of shape
    # (1, n_patches ** 2 + 1)
    def _get_ViT_mask(self, bbox, height, width):
        arr = np.zeros((self.n_patches, self.n_patches))
        x_min, y_min, w, h = bbox
        y_max = y_min + h
        x_max = x_min + w
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

        entries = None
        bbox_info = pd.read_csv(os.path.join(self.config["DATA_FOLDER"], "CXR8", "BBox_List_2017.csv"))
        train_images = set()
        test_images = set()

        
        with open(os.path.join(self.config["DATA_FOLDER"], "CXR8", "test_list.txt"), "r") as f:
            for line in f:
                test_images.add(line.strip())
        with open(os.path.join(self.config["DATA_FOLDER"], "CXR8", "train_val_list.txt"), "r") as f:
            for line in f:
                train_images.add(line.strip())

        final_train = []
        final_test = []
        cur_path_count = {}
        for i in range(len(bbox_info)):
            pathology = bbox_info.loc[i, "Finding Label"]
            if pathology not in cur_path_count:
                cur_path_count[pathology] = 0
            
            if cur_path_count[pathology] < 20:
                final_train.append(bbox_info.loc[i, :])
            else:
                final_test.append(bbox_info.loc[i, :])
            cur_path_count[pathology] += 1
        self.class_counts = cur_path_count
        if self.split == "train":
            entries = final_train
        elif self.split == "test":
            entries = final_test

        final_entries = []
        for entry in entries:
            new_entry = {}
            new_entry["id"] = entry['Image Index'][:-4]
            new_entry["path_to_image"] = os.path.join(os.path.join(self.config["DATA_FOLDER"], "CXR8", "images"), entry['Image Index'])
            new_entry["answer"] = entry["Finding Label"]
            new_entry["bbox"] = [int(entry["Bbox [x"]), int(entry["y"]), int(entry["w"]), int(entry["h]"])]
            image = Image.open(new_entry["path_to_image"])
            new_entry["vit_mask"] = self._get_ViT_mask(new_entry["bbox"], image.height, image.width)
            new_entry["original_segmentation"] = None
            new_entry["question"] = "[obj] What is this?"
            final_entries.append(new_entry)

        return final_entries

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
    # Return: The k closest examples from self.entries according to cosine similarity
    # Note: features should be normalized so dot product == cosine similarity
    def retrieve_closest(self, object_features, k, train_phase = True, b_num=-1):

        dist_matrix = (object_features @ self.retrieval_keys.T)

        if train_phase:
            closest_indices = torch.argsort(dist_matrix, axis = -1, descending=True)[:, 1:1+k]
        else:
            closest_indices = torch.argsort(dist_matrix, axis = -1, descending=True)[:, 0:k]

        similarity_scores = [[round(dist_matrix[i, x].item(), 2) for x in closest_indices[i,:]] for i in range(len(closest_indices))]
        retrieved_info = [[self.entries[x] for x in closest_indices[i,:]] for i in range(len(closest_indices))]


        return retrieved_info, similarity_scores


    def __str__(self):
        return f"CXR8 dataset with {len(self.entries)} questions"


    def __len__(self):
        return len(self.entries)


    def __getitem__(self, index):
        entry = self.entries[index]

        return entry