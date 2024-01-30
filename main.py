import os
import torch
from dataset.RefCOCO import RefCOCODataset
from dataset.GRIT import GRITDataset
from dataset.countCOCO import CountCOCODataset
from dataset.objectCOCO import COCOObjectDataset
from dataset.LVIS import LVISDataset
from dataset.VRP import VRPDataset
from dataset.visualgenome import VisualGenomeDataset
from torch.utils.data import DataLoader
from transformers import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pycocotools.mask as mask
import numpy as np
from tqdm import tqdm
from models.visionllama import VisionLLaMA
from matplotlib.colors import ListedColormap

import pickle
import yaml
import argparse
import os




def main(args):

    logging.set_verbosity_error()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    
    if args.train:
        if config['task'] == 'object_classification':
            dataset = COCOObjectDataset(config, split="train", patch_size=config['patch_size'], max_examples_per_class = config["examples_per_class"])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == "open_vocab_object_classification":
            dataset = LVISDataset(split="val", patch_size=config['patch_size'])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2)
        elif config['task'] == 'image_captioning':
            dataset = VRPDataset("train", patch_size=config['patch_size'], use_object_annotations=config['use_object_annotations'])
            #dataset_val = VRPDataset("val", patch_size=config['patch_size'])
            #dataset_val.entries.extend(dataset_train.entries)
            #dataset = dataset_val
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == 'counting':
            dataset = CountCOCODataset(patch_size=config['patch_size'])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == 'relation_prediction':
            dataset = VisualGenomeDataset(config, split="train", patch_size=config['patch_size'])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == "refCOCO":
            dataset = RefCOCODataset("/data/ossowski/COCO2017/refcoco", "/data/ossowski/COCO2017/train")
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == "refCOCOg":
            dataset = RefCOCODataset("/data/ossowski/COCO2017/refcocog", "/data/ossowski/COCO2017/train", split="train")
            train_loader = DataLoader(dataset, config["batch_size"],  shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == 'ALL':
            dataset1 = RefCOCODataset("/data/ossowski/COCO2017/refcoco", "/data/ossowski/COCO2017/train")
            dataset2 = VRPDataset("train", patch_size=config['patch_size'], use_object_annotations=config['use_object_annotations'])
            dataset3 = COCOObjectDataset(config, split="train", patch_size=config['patch_size'], max_examples_per_class = config["examples_per_class"])
            dataset1.entries.extend(dataset2.entries)
            dataset1.entries.extend(dataset3.entries)
            train_loader = DataLoader(dataset1, config["batch_size"], shuffle=True, num_workers=2, collate_fn=dataset1.collate_fn)

        total = 0
        
        if config["use_retrieval"]:
            model = VisionLLaMA(config, only_use_shape_feature=config["use_only_shape_feature"], retrieval_fn = lambda x: dataset.retrieve_closest(x, config["retrieval_k"]))    
        else:
            model = VisionLLaMA(config, only_use_shape_feature=config["use_only_shape_feature"], retrieval_fn = None)    
        model.prepare_for_training()
        print(f"Model has {model.count_trainable_parameters()} trainable parameters")
        print(f"Model SAVE/LOAD path is {model._get_save_path()}")
        # if config['use_cached_features']:
        #     os.makedirs(config['cache_dir'], exist_ok=True)
        #     cache_path = os.path.join(config['cache_dir'], 'COCO_train.npz')
        #     idx_map_path = os.path.join(config['cache_dir'], 'COCO_idx_map.npz')

        #     if os.path.exists(cache_path) and os.path.exists(idx_map_path):
        #         image_feat_arr = np.load(cache_path, mmap_mode='r')['arr_0']
        #         idx_map = pickle.load(open(idx_map_path, 'rb'))
        #         print(f"Loaded features with shape {image_feat_arr.shape} from {cache_path}")
        #     else:
        #         print("Caching vision features ...")
        #         image_features = []
        #         seen_images = {}
        #         counter = 0
        #         for batch in tqdm(train_loader):
        #             images = batch['path_to_image']
        #             for i in range(len(images)):
        #                 if images[i] in seen_images:
        #                     continue
        #                 seen_images[images[i]] = counter
        #                 pil_image = model.load_image(images[i])
        #                 inputs = model.object_encoder.processor(images=[pil_image], return_tensors="pt").to(config["device"])
        #                 transformer_output = model.object_encoder.model(**inputs, output_attentions=True).last_hidden_state
        #                 image_features.append(transformer_output.detach().cpu().numpy())

        #                 counter += 1
        #         final_features = np.concatenate(image_features, axis = 0)
        #         np.savez(cache_path, final_features)
        #         pickle.dump(seen_images, open(idx_map_path, 'wb'))
        # exit()

        # for batch in tqdm(train_loader):
             
        #     images = batch['path_to_image']
        #     for i in range(len(images)):
        #         if type(images[i]) == list:
        #             images[i] = [model.load_image(x) for x in images[i]]
        #         else:
        #             images[i] = [model.load_image(images[i])]

        #     for i in range(len(images)):
        #         image_input = images[i]

        #         inputs = model.object_encoder.processor(images=image_input, return_tensors="pt").to(config["device"])
            
        #         transformer_output = model.object_encoder.model(pixel_values = inputs['pixel_values']).last_hidden_state

        optimizer = torch.optim.Adam(list(model.parameters()), lr=config['learning_rate'])

        past_losses = []
        best_avg_loss = float("inf")
        finished_training = False
        consecutive_no_improvement = 0
        for epoch in range(config["n_epochs"]):
            model.train()
            if finished_training:
                break
            for b_num, batch in enumerate(tqdm(train_loader)):

                masks = batch["vit_mask"]
                questions = batch['question']
                answers = batch['answer']
                images = batch['path_to_image']
                #print(questions)
                #print(answers)
    
                if type(masks) == list:
                    for i in range(len(masks)):
                        masks[i] = torch.BoolTensor(masks[i]).to(config["device"])
                else:
                    masks = masks.type(torch.BoolTensor)

                

                if config["use_only_shape_feature"]:
                    original_segmentations = batch["original_segmentation"]
                    for i, seg in enumerate(original_segmentations):

                        # seg['size'] = [x[0].item() for x in seg['size']]
                        # seg['counts'] = seg['counts'][0]
                        original_segmentations[i] = mask.decode(seg)

                    #print(type(original_segmentations))
                    if type(original_segmentations) == list:
                        for i in range(len(original_segmentations)):
                            original_segmentations[i] = torch.FloatTensor(original_segmentations[i]).to(config["device"])
                            #print(torch.sum(original_segmentations[0]))
                    else:
                        original_segmentations= torch.FloatTensor(original_segmentations).to(config["device"])


                else:
                    original_segmentations = []


                # with torch.no_grad():
                    
                #     output = model.generate(masks, images.copy(), questions)
                #     print(output)
                

                optimizer.zero_grad()
                #start_time = time.time()
                out = model(masks, images, questions, answers)

                
                #end_time = time.time()
                #print(f"Forward Pass time: {end_time - start_time} seconds")
                if out == None:
                    continue

                loss = out.loss
                #wandb.log({"loss": loss_item})
                past_losses.append(loss.item())
                #print(loss.item())
                #start_time = time.time()
                loss.backward()
                optimizer.step()
                

                del loss
                del out

                torch.cuda.empty_cache()
                #end_time = time.time()
                #print(f"Backward Pass time: {end_time - start_time} seconds")

                if b_num > 0 and b_num % config["check_loss_steps"] == 0:
                    avg_loss = sum(past_losses[-config["check_loss_steps"]:]) / len(past_losses[-config["check_loss_steps"]:])
                    
                    if not config["early_stopping"]:
                        print(f"Loss is {round(avg_loss, 2)}")
                        SAVE_PATH = model._get_save_path()
                        
                        model.save()
                        plt.plot(list(range(len(past_losses))), past_losses)
                        plt.xlabel("Num Parameter Updates")
                        plt.ylabel("Loss")
                        plt.title("Train Loss Graph")
                        plt.savefig(os.path.join(SAVE_PATH, "train.png"))
                        plt.close()
                        continue
                    
                    if avg_loss < best_avg_loss:
                        print(f"{round(avg_loss, 2)} is better than best average loss {round(best_avg_loss, 2)}")
                        best_avg_loss = avg_loss
                        consecutive_no_improvement = 0
                        model.save()
                    else:
                        
                        consecutive_no_improvement += 1
                        if consecutive_no_improvement >= 10 and config["early_stopping"]:
                            updates = 10 * config["check_loss_steps"] * config["batch_size"]
                            print(f"Loss did not improve after {updates} steps, exiting ...")
                            finished_training = True



    elif args.test:
  
        if config['task'] == 'object_classification':
            dataset = COCOObjectDataset(config, split="val", patch_size=config['patch_size'], max_examples_per_class = config["examples_per_class"])
            #train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == "open_vocab_object_classification":
            dataset = LVISDataset(split="val", patch_size=config['patch_size'])
            #train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2)
        elif config['task'] == 'image_captioning':
            dataset = VRPDataset("test", patch_size=config['patch_size'])
            #train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == "refCOCOg":
            dataset = RefCOCODataset("/data/ossowski/COCO2017/refcocog", "/data/ossowski/COCO2017/train", split="val")
        elif config['task'] == 'counting':
            dataset = CountCOCODataset(patch_size=config['patch_size'])
            #train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)


        if config["use_retrieval"]:
            if config['task'] == 'object_classification':
                retrieval_dataset = COCOObjectDataset(config, split="train", patch_size=config['patch_size'], max_examples_per_class = config["examples_per_class"])
            elif config['task'] == "open_vocab_object_classification":
                retrieval_dataset = LVISDataset(split="train", patch_size=config['patch_size'])
            elif config['task'] == 'image_captioning':
                retrieval_dataset = VRPDataset(patch_size=config['patch_size'])
            elif config['task'] == 'counting':
                retrieval_dataset = CountCOCODataset(patch_size=config['patch_size'])
            model = VisionLLaMA(config, only_use_shape_feature=config["use_only_shape_feature"], retrieval_fn = lambda x: retrieval_dataset.retrieve_closest(x, config["retrieval_k"], train_phase=False))    
        else:
            model = VisionLLaMA(config, only_use_shape_feature=config["use_only_shape_feature"], retrieval_fn = None)    

        OUTPUT_SAVE_PATH = os.path.join("outputs", config["task"], model._get_save_path().split("/")[-1] + ".pkl")
        os.makedirs(os.path.join("outputs", config["task"]), exist_ok=True)
        print(OUTPUT_SAVE_PATH)

        correct = 0
        total = 0
        if not os.path.exists(OUTPUT_SAVE_PATH):
            
            json_results = []
            import json
            data = json.load(open("/data/ossowski/COCO2017/annotations/instances_val2017.json"))
            coco_eval_mapping = {c['name']:c['id'] for c in data['categories']}
            
            with open(OUTPUT_SAVE_PATH, "rb") as f:
                data = pickle.load(f)
            if config["task"] == "object_classification":
            
                from sklearn.metrics import recall_score
                from sklearn.metrics import precision_score
                class_names = sorted(dataset.class_counts.keys())
                mapping = {class_names[i]:i for i in range(len(class_names))}
                gt = []
                predictions = []
                #scores = []
                class_amounts = {}
                class_correct = {}
                mistakes = {}
                for key in data:
                    #print(data[key])
                    answer = data[key]["answer"]
                    prediction = data[key]["prediction"]

                    #scores.append(data[key]["confidence"])
                    bbox = dataset.entries[key]['bbox']
                    #print(bbox)
                    d = {}
                    d["image_id"] = int(dataset.entries[key]['id'])
                    d["category_id"] = coco_eval_mapping.get(prediction, 0)
                    d['score'] = 0.99
                    d['bbox'] = dataset.entries[key]['bbox']

                    json_results.append(d)
                    gt.append(mapping[answer])
                    predictions.append(mapping.get(prediction, 0))
                    if answer not in class_amounts:
                        class_amounts[answer] = 0
                        class_correct[answer] = 0
                        mistakes[answer] = {}


                    if answer.lower() ==  prediction.lower():
                        correct += 1
                        class_correct[answer] += 1
                    else:
                        if prediction not in mistakes[answer]:
                            mistakes[answer][prediction] = 0
                        mistakes[answer][prediction] += 1
                    class_amounts[answer] += 1
                    total += 1
                
            out_file = open("/data/ossowski/cocoapi/results/test.json", "w") 
            json.dump(json_results, out_file) 
            print(f"Performance")
            print(f"Overall Accuracy {correct / total:.3}")
            top_5_frequent_classes = sorted(class_amounts, key = lambda x: class_amounts[x], reverse=True)
            print(f"Top 5 Class Performance")
            for i, c in enumerate(top_5_frequent_classes):
                percentage = class_amounts[c] / sum(class_amounts.values())
                performance = class_correct[c] / class_amounts[c]
                most_common_mistake = sorted(mistakes[c].keys(), key = lambda x:mistakes[c][x], reverse=True)[0]
                print(f"{c}: {round(performance, 3)}")
                #print(f"'{c}' class which is {percentage * 100:.3}% of data: {performance:.2}. Most common mistake is {most_common_mistake}") 
            print(f"Overall Accuracy {correct / total:.3}")
            # y_true = [mapping[x] for x in gt]
            # y_pred = [mapping[x] for x in predictions]
            from sklearn.metrics import average_precision_score
            y_scores = np.array([[1 if i == x else 0 for i in range(len(class_names))] for x in gt])
            print(y_scores.shape)
            recall = recall_score(gt, predictions, average='macro')
            precision = precision_score(gt, predictions, average='macro')
            print(precision)
            print(recall)
            print(average_precision_score(np.array(predictions), y_scores))

            exit()

        correct = 0
        total = 0
        model.load()
        model.eval()

        
        #dataset = RefCOCODataset("/data/ossowski/COCO2017/refcoco", "/data/ossowski/COCO2017/train/data", "test")
        #dataset = GRITDataset("/data/ossowski/GRIT")
        #dataset = VRPDataset(split="val")
        train_loader = DataLoader(dataset, 1, shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)

        responses = {}
        from torch.nn.functional import softmax

        with torch.no_grad():
            for i, batch in enumerate(tqdm(train_loader)):
                masks = batch["vit_mask"]

                # for n in range(len(masks)):
                #     masks[n] = torch.BoolTensor(masks[n]).to(config["device"])

                if type(masks) == list:
                    for n in range(len(masks)):
                        masks[n] = torch.BoolTensor(masks[n]).to(config["device"])
                else:
                    masks = masks.type(torch.BoolTensor)

                

                questions = batch['question']
                answers = batch['answer']
                image = batch['path_to_image']
                #example_id = batch["id"][0]
                #print(example_id)

                #questions = [x + "[start]" for x in questions]
                #print(questions)
                # scores = []
                # answer_idx = -1
                # best_class = None
                # best_score = float("inf")
                # for idx, c in enumerate(dataset.stats()):
                #     score = model.get_confidence_score(masks, image.copy(), questions, [c])
                #     if score < best_score:
                #         best_class = c
                #         best_score = score
                #     scores.append(score)
                # print(answers[0], best_class)

                # losses = []
                # best_prediction = None
                # best_score = float("inf")
                # for idx, c in enumerate(dataset.stats()):
                    
                #     out = model(masks, image.copy(), questions, [c], full_segmentations = [])
                #     loss = out.loss.item()
                #     if loss < best_score:
                #         best_score = loss
                #         best_prediction = c
                #         best_idx = idx
                #     losses.append(-loss * 5)
                # confidences = softmax(torch.FloatTensor(losses))
                # confidence = confidences[best_idx]
                #print(confidence, best_prediction, answers[0])
                output = model.generate(masks, image.copy(), questions)
                output = output.lstrip().replace(".", "")
                #print(output)
                # print(output)
                #image = Image.open(image[0])
                # plt.imshow(image)
                # plt.title(output)
                # plt.xlabel(answers)
                #
                # print(output)
                #word = output.split(" ")
                #predicted_indices = [int(x) for x in output.split(" ") if x.isdigit()][1:]
                #print(predicted_indices)
               # print(batch['original_segmentations'])
                #original_segs = batch['original_segmentations']
                #print(len(original_segs))
                #alpha = 0.5
                # if len(original_segs) != 0:
                #     for x in predicted_indices:
                #         seg = original_segs[x]
                #         seg['size'] = [x[0].item() for x in seg['size']]
                #         seg['counts'] = seg['counts'][0]
                #         seg = mask.decode(seg)
                #         seg = np.ma.masked_where(seg == 0, seg)
                        #masked_image = np.ma.masked_where(seg == 0, masks)
                        #print(seg)
                        #cmap = ListedColormap([(0,1,0) + (alpha,)])
                        #plt.imshow(seg, cmap='jet', alpha=alpha)
                #plt.savefig(f"outputs/testing_{i}")
                
                #info_dict = dataset.eval_correctness(output, answers[0])
                #if config['task'] == 'counting':
                #    n_obj_correct += info_dict['num_object_correct']
                    
                #correct += info_dict["score"]
                #print(correct)
                    #print(answers[0], output)

                    
                total += 1
                responses[i] = {}
                responses[i]["path_to_image"] = batch["path_to_image"][0]
                responses[i]["id"] = batch["id"][0]

                responses[i]["question"] = batch['question'][0]
                responses[i]["answer"] = batch['answer'][0]
                responses[i]["prediction"] = output
                #responses[i]["confidence"] = confidence.item()

                print(responses[i])

                #print(f"Accuracy: {correct/total:.2}")
                #print(f"n_score: {n_obj_correct/total:.2}")

                # fig, ax = plt.subplots()

            
            print(round(correct / total, 2))
        with open(OUTPUT_SAVE_PATH, "wb") as f:
            print(f"Saved model outputs to {OUTPUT_SAVE_PATH}")
            pickle.dump(responses, f)


    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/config.yaml")
    # parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--image-file", type=str, required=True)
    # parser.add_argument("--num-gpus", type=int, default=1)
    # parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    # parser.add_argument("--conv-mode", type=str, default=None)
    # parser.add_argument("--temperature", type=float, default=0.2)
    # parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")

    # parser.add_argument("--load-4bit", action="store_true")
    # parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(args)

    # with cProfile.Profile() as profile:
    #     main(args)
    # results = pstats.Stats(profile)
    # results.sort_stats(pstats.SortKey.CUMULATIVE)
    # results.print_stats()

    
