import os
import torch

from dataset.RefCOCO import RefCOCODataset
from dataset.countCOCO import CountCOCODataset
from dataset.objectCOCO import COCOObjectDataset
from dataset.LVIS import LVISDataset
from dataset.VRP import VRPDataset
from dataset.CXR8 import CXR8Dataset
from dataset.customRetrieval import RetrievalDataset
from torch.utils.data import DataLoader
from transformers import logging, CLIPImageProcessor, CLIPVisionModel, AutoImageProcessor, AutoModel
from PIL import Image

import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
import pickle
import yaml
import argparse
import os
from collections import Counter
from pycocotools import mask as _mask
from matplotlib import pyplot as plt

import json


# Gets an object feature based on ViT masks of shape (1, n_patches ** 2 + 1)
# Feature is obtained by taking the mean of the patch features of the object
def get_object_feature(batch, processor, model, config):
    masks = batch["vit_mask"]
    images = batch['path_to_image']
    original_seg = batch["original_segmentation"][0]
    bbox = batch["bbox"][0]

    if config["crop_image"]:
        path_to_image = images[0]
        img = np.array(Image.open(path_to_image).convert('RGB'))
        if original_seg != None:
            m = _mask.decode(original_seg).astype(bool)
            img[~m] = np.array([255,255,255])
        bbox = [int(x) for x in bbox]
        img = img[bbox[1]: bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        
        h, w, _ = img.shape
        if h < 2 or w < 2:
            return None
        images = [Image.fromarray(np.uint8(img)).convert('RGB')]

    else:
        images = [Image.open(x).convert('RGB') for x in images]

    inputs = processor(images=images, return_tensors="pt").to(config["device"])
    image_forward_outs = model(inputs['pixel_values'], output_hidden_states=True)
    mask = torch.BoolTensor(masks[0][1:]).to(config["device"])

    if config["crop_image"]:
        object_feature = image_forward_outs.hidden_states[-1][0,0,:].detach().cpu()
    else:
        image_features_high = image_forward_outs.hidden_states[-1][0,1:,:][mask].detach().cpu()
        object_feature = torch.mean(image_features_high, dim = 0)

    object_feature /= object_feature.norm(dim=-1, keepdim=True)
    return object_feature

def main(args):

    logging.set_verbosity_error()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    DATA_FOLDER = config["DATA_FOLDER"]
    config['n_patches'] = 16
    if "336" in config['vision_encoder']:
        config['n_patches'] = 24
        
    config["batch_size"] = 1

    if config["crop_image"]:
        cropped = "cropped_"
    else:
        cropped = ""

    if "clip" in config["vision_encoder"]:
        model = CLIPVisionModel.from_pretrained(config["vision_encoder"]).to(config["device"])
        processor = CLIPImageProcessor.from_pretrained(config["vision_encoder"])
    elif "dino" in config["vision_encoder"]:
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
        model = AutoModel.from_pretrained('facebook/dinov2-large').to(config["device"])
        model.eval()

    if args.train:
        
        object_feats = []
        labels = []
        image_ids = []
        retrieval_index = []
        retrieval_path = f'retrieval/{config["task"]}/retrieval_set_{config["examples_per_class"]}_{cropped}{config["vision_encoder"].split("/")[-1]}.pkl'
        
        print("Loading Dataset to Create Retrieval Features From ...")
        if "additional_retrieval_examples" in config:
            retrieval_path = os.path.join(config["additional_retrieval_examples"], f'{cropped}{config["vision_encoder"].split("/")[-1]}' + ".pkl")
            dataset = RetrievalDataset(config)
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)

        elif config['task'] == 'object_classification':
            dataset = COCOObjectDataset(config, split="train", n_patches=config['n_patches'], max_examples_per_class = config["examples_per_class"])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == 'medical_object_classification':
            dataset = CXR8Dataset(config, split="train", n_patches=config['n_patches'])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == "open_vocab_object_classification":
            dataset = LVISDataset(config, split="train", n_patches=config['n_patches'])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == "refCOCOg":
            dataset = RefCOCODataset("/data/ossowski/COCO2017/refcocog", "/data/ossowski/COCO2017/train")
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == 'image_captioning':
            dataset = VRPDataset(n_patches=config['n_patches'])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == 'counting':
            dataset = CountCOCODataset(n_patches=config['n_patches'])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        

        print(f'The retrieval features will be saved in {retrieval_path}')
        if config["crop_image"]:
            print("Will crop objects to create the feature")

        for b_num, batch in enumerate(tqdm(train_loader)):
            answers = batch['answer']
            image_id = batch['id']

            object_feature = get_object_feature(batch, processor, model, config)
            if object_feature == None:
                continue
            if len(torch.nonzero(torch.isnan(object_feature.view(-1)))) > 0:
                print(object_feature)
                exit()

            object_feats.append(object_feature)
            labels.append(answers[0])
            image_ids.append(image_id[0])
            retrieval_index.append(b_num)

        retrieval_set = {}
        retrieval_set['keys'] = np.stack(object_feats, axis = 0)
        retrieval_set['values'] = labels
        retrieval_set['idx'] = retrieval_index

        os.makedirs(f"retrieval/{config['task']}", exist_ok=True)
        with open(retrieval_path, 'wb') as f:
            print(retrieval_path)
            pickle.dump(retrieval_set, f)

    elif args.test:
        from collections import Counter
        config["batch_size"] = 1
        cache = {} # Saving most similar objects
        CACHE_PATH = f'cache/{config["task"]}/retrieval_cache_{config["examples_per_class"]}_{cropped}{config["vision_encoder"].split("/")[-1]}.pkl'
        print(CACHE_PATH)
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, 'rb') as f:
                try:
                    cache = pickle.load(f)
                    print(f"Loaded cached retrieval similarity from {CACHE_PATH}")
                except:
                    cache = {}
                    print(f"Problem Loading Cache (dictionary empty)")
                

        if config['task'] == 'object_classification':
            dataset = COCOObjectDataset(config, split="val", n_patches=config['n_patches'], max_examples_per_class = 1000000)
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == 'medical_object_classification':
            dataset = CXR8Dataset(config, split="test", n_patches=config['n_patches'])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == "open_vocab_object_classification":
            dataset = LVISDataset(config, split="val", n_patches=config['n_patches'])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == 'image_captioning':
            dataset = VRPDataset(n_patches=config['n_patches'])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == 'counting':
            dataset = CountCOCODataset(n_patches=config['n_patches'])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)

        with open(f'retrieval/{config["task"]}/retrieval_set_{config["examples_per_class"]}_{cropped}{config["vision_encoder"].split("/")[-1]}.pkl', 'rb') as f:
            data = pickle.load(f)

        keys = data['keys']
        labels = data['values']
        
        top_1 = 0
        top_5 = 0
        top_10 = 0
        majority_correct = 0
        correct_by_category = {}
        majority_correct_by_category = {}
        total_by_category = {}

        data = json.load(open(os.path.join(DATA_FOLDER, "COCO2017", "annotations", "instances_val2017.json")))
        coco_eval_mapping = {c['name']:c['id'] for c in data['categories']}
        json_results = []
        dataset.stats()

        for b_num, batch in enumerate(tqdm(train_loader)):
  
            answers = batch['answer']
            image_id = batch['id'][0]
            bbox = batch['bbox'][0]
    
            if b_num not in cache:
                object_feature = get_object_feature(batch, processor, model, config)
                if object_feature == None:
                    continue
                object_feature = object_feature.unsqueeze(0)

                key = object_feature.detach().cpu().numpy()
                closest = np.dot(key, keys.T)
                indices = list(range(len(closest[0])))
                c = sorted(indices, key = lambda x: closest[0][x], reverse=True)[:50]
                cache[b_num] = [(x,closest[0][x]) for x in c]
            else:
                c = [x[0] for x in cache[b_num]]
            
            predictions = [labels[x] for x in c]
            
            
            k = config["retrieval_k"]
            majority_element = max(Counter(predictions[:k]), key=Counter(predictions[:k]).get)
            if majority_element == answers[0]:
                if answers[0] not in majority_correct_by_category:
                    majority_correct_by_category[answers[0]] = 0
                majority_correct_by_category[answers[0]] += 1
                majority_correct += 1

            if answers[0] in predictions[:10]:
                if answers[0] not in correct_by_category:
                    correct_by_category[answers[0]] = 0
                top_10 += 1
            if answers[0] in predictions[:5]:
                top_5 += 1
            if answers[0] == predictions[0]:
                correct_by_category[answers[0]] += 1
                top_1 += 1

            if answers[0] not in total_by_category:
                total_by_category[answers[0]] = 0
            total_by_category[answers[0]] += 1
            
            d = {}
            d["image_id"] = int(image_id)
            d["category_id"] = coco_eval_mapping.get(majority_element, 0)
            d['score'] = 0.99
            d['bbox'] = bbox

            json_results.append(d)
                
        for category in sorted(correct_by_category, key = lambda x: total_by_category[x], reverse=True):
            print(f"{category}: {majority_correct_by_category.get(category, 0)/ total_by_category[category]:.3f}")
        

        print(f"Top 1: {top_1 / len(dataset)}")
        print(f"Top 5: {top_5 / len(dataset)}")
        print(f"Top 10: {top_10 / len(dataset)}")
        print(f"Majority Correct ({k}): {majority_correct / len(dataset):.5}")


        os.makedirs("outputs", exist_ok=True)
        out_file = open(os.path.join("outputs", "prediction_results.json"), "w") 
        json.dump(json_results, out_file) 


        performances = {}
        for i in range(1,51):
            majority_correct = 0
            for b_num, batch in enumerate(train_loader):
                answers = batch['answer']
                c = [x[0] for x in cache[b_num]]
                predictions = [labels[x] for x in c]
                from collections import Counter
                
                majority_element = max(Counter(predictions[:i]), key=Counter(predictions[:i]).get)
                if majority_element == answers[0]:
                    majority_correct += 1
        
            print(f"Majority Correct ({i}): {majority_correct / len(dataset):.2}")
            performances[i] = round(majority_correct / len(dataset), 4)
        print(performances)

        # Cache results so we don't have to compute retrieval similarity again
        os.makedirs(f"cache/{config['task']}", exist_ok=True)
        with open(f'cache/{config["task"]}/retrieval_cache_{config["examples_per_class"]}_{cropped}{config["vision_encoder"].split("/")[-1]}.pkl', 'wb') as f:
            pickle.dump(cache, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/config.yaml")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    main(args)


    
