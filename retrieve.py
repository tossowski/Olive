import os
import torch
from dataset.RefCOCO import RefCOCODataset
from dataset.GRIT import GRITDataset
from dataset.countCOCO import CountCOCODataset
from dataset.objectCOCO import COCOObjectDataset
from dataset.LVIS import LVISDataset
from dataset.VRP import VRPDataset
from torch.utils.data import DataLoader
from transformers import logging, CLIPModel, CLIPImageProcessor, CLIPVisionModel, AutoImageProcessor, AutoModel
from PIL import Image

import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
import pickle
import yaml
import argparse
import requests
import os

from collections import Counter


def load_image(image_file):

    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def main(args):

    logging.set_verbosity_error()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    config["batch_size"] = 1
    cropped = "cropped_" if config["crop_image"] else ""
    total = 0
    from pycocotools import mask as _mask

    if "clip" in config["vision_encoder"]:
        model = CLIPVisionModel.from_pretrained(config["vision_encoder"]).to(config["device"])
        model.eval()
        processor = CLIPImageProcessor.from_pretrained(config["vision_encoder"])
    elif "dino" in config["vision_encoder"]:
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
        model = AutoModel.from_pretrained('facebook/dinov2-large').to(config["device"])
        model.eval()

    if args.train:

        if config['task'] == 'object_classification':
            dataset = COCOObjectDataset(config, split="train", patch_size=config['patch_size'], max_examples_per_class = config["examples_per_class"])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == "open_vocab_object_classification":
            dataset = LVISDataset(split="train", patch_size=config['patch_size'])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == 'image_captioning':
            dataset = VRPDataset(patch_size=config['patch_size'])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == 'counting':
            dataset = CountCOCODataset(patch_size=config['patch_size'])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)

        
        print(dataset.stats())
        object_feats = []
        labels = []
        image_ids = []
        all_masks = []
        for b_num, batch in enumerate(tqdm(train_loader)):

            masks = batch["vit_mask"]
            questions = batch['question']
            answers = batch['answer']
            images = batch['path_to_image']
            image_id = batch['id']
            bbox = batch['bbox'][0]
            original_seg = batch['original_segmentation'][0]
            image_ids.append(image_id[0])
            all_masks.append(masks[0])

            if config["crop_image"]:
                path_to_image = images[0]
                img = np.array(Image.open(path_to_image).convert('RGB'))
                m = _mask.decode(original_seg).astype(bool)
                # print(mask.shape)
                # print(img.shape)
                img[~m] = np.array([255,255,255])
                bbox = [int(x) for x in bbox]
                img = img[bbox[1]: bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
                h, w, c = img.shape
                if h < 16 or w < 16:
                    continue
                images = [Image.fromarray(np.uint8(img)).convert('RGB')]
            else:
                images = [load_image(x) for x in images]
      

            # if type(masks) == list:
            #     for i in range(len(masks)):
            #         masks[i] = torch.BoolTensor(masks[i]).to(config["device"])
            # else:
            #     masks = masks.type(torch.BoolTensor)

            #print(images)

            inputs = processor(images=images, return_tensors="pt").to(config["device"])
            #print(inputs)
            image_forward_outs = model(inputs['pixel_values'], output_hidden_states=True)

            mask = torch.BoolTensor(masks[0][1:]).to(config["device"])

            if config["crop_image"]:
                object_feature = image_forward_outs.hidden_states[-1][0,0,:].detach().cpu()
            else:
                image_features_high = image_forward_outs.hidden_states[-1][0,1:,:][mask].detach().cpu()
                object_feature = torch.mean(image_features_high, dim = 0)

            object_feature /= object_feature.norm(dim=-1, keepdim=True)
            if len(torch.nonzero(torch.isnan(object_feature.view(-1)))) > 0:
                print(object_feature)
                print(mask)
                exit()
            object_feats.append(object_feature)
            labels.append(answers[0])

            # del inputs
            # torch.cuda.empty_cache()

            # Get mean representation of object
            #transformer_output[i][masks[0]]
        retrieval_set = {}
        retrieval_set['keys'] = np.stack(object_feats, axis = 0)
        retrieval_set['values'] = labels
        retrieval_set['id'] = image_ids
        
        
        with open(f'retrieval/{config["task"]}/retrieval_set_{config["examples_per_class"]}_{cropped}{config["vision_encoder"].split("/")[-1]}.pkl', 'wb') as f:
            pickle.dump(retrieval_set, f)

    elif args.test:
        


        config["batch_size"] = 1
        cache = {} # Saving most similar objects
        CACHE_PATH = f'cache/{config["task"]}/retrieval_cache_{config["examples_per_class"]}_{cropped}{config["vision_encoder"].split("/")[-1]}.pkl'
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, 'rb') as f:
                cache = pickle.load(f)
                print(f"Loaded cached retrieval similarity from {CACHE_PATH}")

        if config['task'] == 'object_classification':
            dataset = COCOObjectDataset(config, split="val", patch_size=config['patch_size'], max_examples_per_class = 1000000)
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == "open_vocab_object_classification":
            dataset = LVISDataset(split="val", patch_size=config['patch_size'])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == 'image_captioning':
            dataset = VRPDataset(patch_size=config['patch_size'])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == 'counting':
            dataset = CountCOCODataset(patch_size=config['patch_size'])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)

        with open(f'retrieval/{config["task"]}/retrieval_set_{config["examples_per_class"]}_{cropped}{config["vision_encoder"].split("/")[-1]}.pkl', 'rb') as f:
            data = pickle.load(f)
        
        from transformers import CLIPProcessor, CLIPModel

        #text_model = CLIPModel.from_pretrained(config["vision_encoder"]).to(config["device"])
        #text_processor = CLIPProcessor.from_pretrained(config["vision_encoder"])
        #class_names = list(dataset.class_counts.keys())
        #inputs = text_processor(text=[f"a photo of a {c}" for c in class_names], images=None, return_tensors="pt", padding=True)
        #text_features = text_model.get_text_features(torch.tensor(inputs.input_ids, dtype=torch.long).to(config["device"]))
        #print(text_features.shape)
        #text_features /= text_features.norm(dim=-1, keepdim=True)
        #text_features = text_features.to(config["device"])

        keys = data['keys']
        labels = data['values']
        
        top_1 = 0
        top_5 = 0
        top_10 = 0
        majority_correct = 0
        correct_by_category = {}
        majority_correct_by_category = {}
        total_by_category = {}

        import json
        data = json.load(open("/data/ossowski/COCO2017/annotations/instances_val2017.json"))
        coco_eval_mapping = {c['name']:c['id'] for c in data['categories']}
        json_results = []
        for b_num, batch in enumerate(tqdm(train_loader)):
  
            masks = batch["vit_mask"]
            questions = batch['question']
            answers = batch['answer']
            images = batch['path_to_image']
            image_id = batch['id'][0]
            bbox = batch['bbox'][0]
  
            
            if b_num not in cache:

                
                if config["crop_image"]:
                    path_to_image = images[0]
                    img = np.array(Image.open(path_to_image))
                    m = mask.decode(original_seg).astype(bool)
                    # print(mask.shape)
                    # print(img.shape)
                    img[~m] = np.array([255,255,255])
                    bbox = [int(x) for x in bbox]
                    img = img[bbox[1]: bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
                    h, w, c = img.shape
                    if h < 2 or w < 2:
                        continue
                    images = [Image.fromarray(np.uint8(img)).convert('RGB')]
                else:
                    images = [load_image(x) for x in images]

                inputs = processor(images=images, return_tensors="pt").to(config["device"])
                image_forward_outs = model(inputs['pixel_values'], output_hidden_states=True)
                
                #projected_image_hidden_states = text_model.visual_projection(image_forward_outs.hidden_states[-1])
                #print(image_forward_outs.hidden_states[-1].shape)
                mask = torch.BoolTensor(masks[0][1:]).to(config["device"])

                if config["crop_image"]:
                    object_feature = image_forward_outs.hidden_states[-1][0,0,:].detach().cpu()
                else:
                    image_features_high = image_forward_outs.hidden_states[-1][0,1:,:][mask].detach().cpu()
                    object_feature = torch.mean(image_features_high, dim = 0)

                object_feature /= object_feature.norm(dim=-1, keepdim=True)
                object_feature = object_feature.unsqueeze(0)

                #similarity = (100.0 * object_feature @ text_features.T).softmax(dim=-1)
                #values, indices = similarity[0].topk(5)
                #print(values, [class_names[idx] for idx in indices], answers[0])
                #continue

                key = object_feature.detach().cpu().numpy()
                #print(np.linalg.norm(keys, axis = 1))
                #closest = cdist(key, keys, metric='cosine')
                closest = np.dot(key, keys.T)
                #print(closest)
                #break
                indices = list(range(len(closest[0])))
                c = sorted(indices, key = lambda x: closest[0][x], reverse=True)[:50]
                cache[b_num] = [(x,closest[0][x]) for x in c]
            else:
                c = [x[0] for x in cache[b_num]]
            #print([closest[0][x] for x in c], answers[0])
            # torch.cuda.empty_cache()
            

            predictions = [labels[x] for x in c]
            
            #print(predictions, answers[0])
            from collections import Counter
            
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
                
            
                
            #MAP Calc Score
            # gt_boxes.append([b_num, b_num, b_num+1, b_num+1])
            # pred_boxes.append([b_num, b_num, b_num+1, b_num+1])
            # obj_name = answers[0]
            # gt_labels.append(obj_name_to_id[obj_name])
            # pred_labels.append(obj_name_to_id.get(majority_element))
            # scores.append(1)

            # ground_truth = {
            #     'boxes':
            #         gt_boxes,

            #     'labels':
            #         gt_labels}

            # result_dict = {
            #     'boxes':
            #         pred_boxes,

            #     'labels':
            #         pred_labels,
                    
            #     'scores': scores}
            # print(calculate_map(ground_truth, result_dict, 0.5))
        # counts = Counter(labels)
        # for v in sorted(counts, key = lambda x: counts[x]):
        #     print(v, counts[v], counts[v] / sum(counts.values()))

        for category in sorted(correct_by_category, key = lambda x: total_by_category[x], reverse=True):
            #print(f"{category}: {correct_by_category[category] / total_by_category[category]:.3f}")
            print(f"{category}: {majority_correct_by_category.get(category, 0)/ total_by_category[category]:.3f}")
        

        print(f"Top 1: {top_1 / len(dataset)}")
        print(f"Top 5: {top_5 / len(dataset)}")
        print(f"Top 10: {top_10 / len(dataset)}")
        print(f"Majority Correct ({k}): {majority_correct / len(dataset):.5}")
        out_file = open("/data/ossowski/cocoapi/results/test.json", "w") 
        json.dump(json_results, out_file) 
        # performances = {}
        # for i in range(1,51):
        #     majority_correct = 0
        #     for b_num, batch in enumerate(train_loader):
        #         answers = batch['answer']
        #         c = [x[0] for x in cache[b_num]]
        #         predictions = [labels[x] for x in c]
        #         from collections import Counter
                
        #         majority_element = max(Counter(predictions[:i]), key=Counter(predictions[:i]).get)
        #         if majority_element == answers[0]:
        #             majority_correct += 1
        
        #     print(f"Majority Correct ({i}): {majority_correct / len(dataset):.2}")
        #     performances[i] = round(majority_correct / len(dataset), 4)
        # print(performances)



        os.makedirs(f"cache/{config['task']}", exist_ok=True)
        with open(f'cache/{config["task"]}/retrieval_cache_{config["examples_per_class"]}_{config["feature_select_layer"]}_{config["vision_encoder"].split("/")[-1]}.pkl', 'wb') as f:
            pickle.dump(cache, f)



            


            # Get mean representation of object
            #transformer_output[i][masks[0]]
       
   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/config.yaml")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    main(args)


    
