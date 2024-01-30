import os
import torch
from dataset.RefCOCO import RefCOCODataset
from dataset.countCOCO import CountCOCODataset
from dataset.objectCOCO import COCOObjectDataset
from dataset.LVIS import LVISDataset
from dataset.VRP import VRPDataset
from dataset.visualgenome import VisualGenomeDataset
from torch.utils.data import DataLoader
from transformers import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.visionllama import VisionLLaMA

import pickle
import yaml
import argparse
import os


def main(args):

    logging.set_verbosity_error()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    
    if args.train:

        print("Loading Dataset ...")
        if config['task'] == 'object_classification':
            dataset = COCOObjectDataset(config, split="train", patch_size=config['patch_size'], max_examples_per_class = config["examples_per_class"])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == "open_vocab_object_classification":
            dataset = LVISDataset(split="val", patch_size=config['patch_size'])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2)
        elif config['task'] == 'image_captioning':
            dataset = VRPDataset("train", patch_size=config['patch_size'], use_object_annotations=config['use_object_annotations'])
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
            print(f"Using Retrieval with k = {config['retrieval_k']}")
            model = VisionLLaMA(config, retrieval_fn = lambda x: dataset.retrieve_closest(x, config["retrieval_k"]))    
        else:
            model = VisionLLaMA(config, retrieval_fn = None)   

        print(f"Model SAVE/LOAD path is {model._get_save_path()}") 
        model.prepare_for_training()

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

                if type(masks) == list:
                    for i in range(len(masks)):
                        masks[i] = torch.BoolTensor(masks[i]).to(config["device"])
                else:
                    masks = masks.type(torch.BoolTensor)
                

                optimizer.zero_grad()
                out = model(masks, images, questions, answers)
                if out == None:
                    continue

                loss = out.loss
                past_losses.append(loss.item())
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
        elif config['task'] == "open_vocab_object_classification":
            dataset = LVISDataset(split="val", patch_size=config['patch_size'])
        elif config['task'] == 'image_captioning':
            dataset = VRPDataset("test", patch_size=config['patch_size'])
        elif config['task'] == "refCOCOg":
            dataset = RefCOCODataset("/data/ossowski/COCO2017/refcocog", "/data/ossowski/COCO2017/train", split="val")
        elif config['task'] == 'counting':
            dataset = CountCOCODataset(patch_size=config['patch_size'])


        if config["use_retrieval"]:
            if config['task'] == 'object_classification':
                retrieval_dataset = COCOObjectDataset(config, split="train", patch_size=config['patch_size'], max_examples_per_class = config["examples_per_class"])
            elif config['task'] == "open_vocab_object_classification":
                retrieval_dataset = LVISDataset(split="train", patch_size=config['patch_size'])
            elif config['task'] == 'image_captioning':
                retrieval_dataset = VRPDataset(patch_size=config['patch_size'])
            elif config['task'] == 'counting':
                retrieval_dataset = CountCOCODataset(patch_size=config['patch_size'])
            model = VisionLLaMA(config, retrieval_fn = lambda x: retrieval_dataset.retrieve_closest(x, config["retrieval_k"], train_phase=False))    
        else:
            model = VisionLLaMA(config, retrieval_fn = None)    

        OUTPUT_SAVE_PATH = os.path.join("outputs", config["task"], model._get_save_path().split("/")[-1] + ".pkl")
        os.makedirs(os.path.join("outputs", config["task"]), exist_ok=True)
        print(OUTPUT_SAVE_PATH)

        correct = 0
        total = 0
        if os.path.exists(OUTPUT_SAVE_PATH):
            
            json_results = []
            import json
            data = json.load(open("/data/ossowski/COCO2017/annotations/instances_val2017.json"))
            coco_eval_mapping = {c['name']:c['id'] for c in data['categories']}
            
            with open(OUTPUT_SAVE_PATH, "rb") as f:
                data = pickle.load(f)
            if config["task"] == "object_classification":

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

            exit()

        model.load()
        model.eval()

        train_loader = DataLoader(dataset, 1, shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)

        responses = {}

        with torch.no_grad():
            for i, batch in enumerate(tqdm(train_loader)):
                masks = batch["vit_mask"]

                if type(masks) == list:
                    for n in range(len(masks)):
                        masks[n] = torch.BoolTensor(masks[n]).to(config["device"])
                else:
                    masks = masks.type(torch.BoolTensor)

                questions = batch['question']
                answers = batch['answer']
                image = batch['path_to_image']
       
                output = model.generate(masks, image.copy(), questions)
                output = output.lower().lstrip().rstrip().replace(".", "")
                    
                responses[i] = {}
                responses[i]["path_to_image"] = batch["path_to_image"][0]
                responses[i]["id"] = batch["id"][0]

                responses[i]["question"] = batch['question'][0]
                responses[i]["answer"] = batch['answer'][0]
                responses[i]["prediction"] = output
                #responses[i]["confidence"] = confidence.item()

                print(responses[i])

        with open(OUTPUT_SAVE_PATH, "wb") as f:
            print(f"Saved model outputs to {OUTPUT_SAVE_PATH}")
            pickle.dump(responses, f)


    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/config.yaml")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    main(args)



    
