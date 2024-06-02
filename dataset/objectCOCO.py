import json
import os
import numpy as np
import pycocotools.mask as mask
import torch
import pickle
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from dataset.customRetrieval import RetrievalDataset

# Object instruction following dataset for referring object classification
class COCOObjectDataset(Dataset):
    def __init__(self, config, split="train", n_patches=16, max_examples_per_class = 1000000):
        super(COCOObjectDataset, self).__init__()
        self.split = split
        if "train" in split:
            dataset_path = os.path.join(f"{config['DATA_FOLDER']}", "COCO2017", "instruction_data", f"supervised_with_segmentations_train_object_detection_{n_patches}x{n_patches}.json")
        else:
            dataset_path = os.path.join(f"{config['DATA_FOLDER']}", "COCO2017", "instruction_data", f"supervised_with_segmentations_val_object_detection_{n_patches}x{n_patches}.json")

        self.config = config
        self.data = json.load(open(dataset_path))
        self.prompts = ["[obj] What is this? Answer with a short word or phrase.",
                        "[obj] What is this object? Answer in 1-2 words.",
                        "Here is an object [obj]. What is this? Answer with a short word or phrase."]
        self.n_patches = n_patches
        self.max_examples_per_class = max_examples_per_class
        self.class_counts = {}
        self.entries = self._load_dataset()

        if config["use_retrieval"]:
            if config["crop_image"]:
                cropped = "cropped_"
            else:
                cropped = ""


            if "retrieval_set_path" in config:
                retrieval_path = config["retrieval_set_path"]
            else:
                retrieval_path = f'retrieval/{config["task"]}/retrieval_set_{config["examples_per_class"]}_{cropped}{config["vision_encoder"].split("/")[-1]}.pkl'

            with open(retrieval_path, 'rb') as f:
                self.retrieval_data = pickle.load(f)
                self.retrieval_keys = torch.FloatTensor(self.retrieval_data['keys']).to(self.config["device"])
                self.retrieval_labels = self.retrieval_data['values']
                self.retrieval_idx = self.retrieval_data.get('idx', None)
                assert len(self.retrieval_keys) == len(self.retrieval_labels)
                print(f'Loaded {len(self.retrieval_keys)} examples from {retrieval_path}')

            # Augment data with additional retrieval examples if it is in the config
            if "additional_retrieval_examples" in config:
                path_to_additional_examples = os.path.join(config["additional_retrieval_examples"], f'{cropped}{config["vision_encoder"].split("/")[-1]}' + ".pkl")
                dataset = RetrievalDataset(config)
                with open(path_to_additional_examples, "rb") as f:
                    data = pickle.load(f)
                self.additional_retrieval_keys = torch.FloatTensor(data['keys']).to(self.config["device"])
                self.additional_retrieval_labels = data['values']

                if self.retrieval_idx:
                    self.additional_retrieval_idx = data.get('idx', None)
                    self.additional_retrieval_idx = [x + len(self) for x in self.additional_retrieval_idx]
                    self.retrieval_idx.extend(self.additional_retrieval_idx)
                self.retrieval_labels.extend(self.additional_retrieval_labels)
                self.retrieval_keys = torch.cat((self.retrieval_keys, self.additional_retrieval_keys), axis = 0)
                self.entries.extend(dataset.entries)


                
    
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
                seg = np.append(1, mask.decode(segmentation).flatten())

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
                chunk.append({"id": item["id"], "path_to_image": item["image"], "question": random.choice(self.prompts), "vit_mask": seg, "answer": label, "original_segmentation": original_seg, 'bbox':bbox})
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

        return class_counts

    # Should take in argument k: how many closest objects to retrieve
    # and object features: the ViT features of query objects
    # Return: The k closest examples from self.entries according to cosine similarity
    # Note: features should be normalized so dot product == cosine similarity
    def retrieve_closest(self, object_features, k, train_phase = True, b_num=-1):
 
        dist_matrix = (object_features @ self.retrieval_keys.T)

        # If training, do not retrieve closest object to avoid label leakage
        if train_phase:
            closest_indices = torch.argsort(dist_matrix, axis = -1, descending=True)[:, 1:1+k]
        else:
            closest_indices = torch.argsort(dist_matrix, axis = -1, descending=True)[:, 0:k]


        similarity_scores = [[round(dist_matrix[i, x].item(), 2) for x in closest_indices[i,:]] for i in range(len(closest_indices))]
        
        # If there is retrieval idx, this means some objects could not be encoded during retrieval
        # set construction (see retrieve.py)
        # So there isn't a 1-1 correspondence between self.entries and retrieval keys
        # That's why there is a retrieval idx, where index i in this self.retrieval_idx
        # points to the corresponding index in self.entries
        if self.retrieval_idx: 
            retrieved_info = [[self.entries[self.retrieval_idx[x]] for x in closest_indices[i,:]] for i in range(len(closest_indices))]
        else:
            retrieved_info = [[self.entries[x] for x in closest_indices[i,:]] for i in range(len(closest_indices))]
            
        return retrieved_info, similarity_scores


    def __str__(self):
        return f"ObjectCOCO dataset with {len(self.entries)} questions"


    def __len__(self):
        return len(self.entries)


    def __getitem__(self, index):
        entry = self.entries[index]

        return entry