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
            dataset = COCOObjectDataset(config, split="val", patch_size=config['patch_size'], max_examples_per_class = config["examples_per_class"])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == "open_vocab_object_classification":
            dataset = LVISDataset(config, split="val", patch_size=config['patch_size'])
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
            dataset2 = RefCOCODataset("/data/ossowski/COCO2017/refcocog", "/data/ossowski/COCO2017/train", patch_size=config["patch_size"], split="train")
            dataset1 = VRPDataset("train", patch_size=config['patch_size'], use_object_annotations=config['use_object_annotations'])
            dataset3 = COCOObjectDataset(config, split="train", patch_size=config['patch_size'], max_examples_per_class = config["examples_per_class"])
            dataset1.entries.extend(dataset2.entries)
            dataset1.entries.extend(dataset3.entries)
            train_loader = DataLoader(dataset1, config["batch_size"], shuffle=True, num_workers=2, collate_fn=dataset1.collate_fn)

        total = 0
        
        if config["use_retrieval"]:
            print(f"Using Retrieval with k = {config['retrieval_k']}")
            model = VisionLLaMA(config, retrieval_fn = lambda x, b: dataset.retrieve_closest(x, config["retrieval_k"], b_num=b))    
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

                        period = 100
                        averages = []
                        for i in range(period, len(past_losses)):
                            subset = past_losses[i-period:i]
                            averages.append(sum(subset) / len(subset))
                            
                        plt.plot(list(range(len(averages))), averages)
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
        from eval.utils import eval_captioning, eval_object_classification
        print("Loading Validation Dataset ...")
        if config['task'] == 'object_classification':
            dataset = COCOObjectDataset(config, split="val", patch_size=config['patch_size'], max_examples_per_class = config["examples_per_class"])
        elif config['task'] == "open_vocab_object_classification":
            dataset = LVISDataset(config, split="val", patch_size=config['patch_size'])
        elif config['task'] == 'image_captioning':
            dataset = VRPDataset("test", patch_size=config['patch_size'])
        elif config['task'] == "refCOCOg":
            dataset = RefCOCODataset("/data/ossowski/COCO2017/refcocog", "/data/ossowski/COCO2017/train", patch_size=config["patch_size"], split="val")
        elif config['task'] == 'counting':
            dataset = CountCOCODataset(patch_size=config['patch_size'])


        if config["use_retrieval"]:
            print("Preparing Retrieval Dataset from Train Split ...")
            if config['task'] == 'object_classification':
                retrieval_dataset = COCOObjectDataset(config, split="train", patch_size=config['patch_size'], max_examples_per_class = config["examples_per_class"])
            elif config['task'] == "open_vocab_object_classification":
                retrieval_dataset = LVISDataset(config, split="train", patch_size=config['patch_size'])
            elif config['task'] == 'image_captioning':
                retrieval_dataset = VRPDataset(patch_size=config['patch_size'])
            elif config['task'] == 'counting':
                retrieval_dataset = CountCOCODataset(patch_size=config['patch_size'])
            model = VisionLLaMA(config, retrieval_fn = lambda x, b: retrieval_dataset.retrieve_closest(x, config["retrieval_k"], train_phase=False, b_num=b))    
        else:
            model = VisionLLaMA(config, retrieval_fn = None)    

        OUTPUT_SAVE_PATH = os.path.join("outputs", config["task"], model._get_save_path(load_raw=True).split("/")[-1] + ".pkl")
        os.makedirs(os.path.join("outputs", config["task"]), exist_ok=True)
        print(OUTPUT_SAVE_PATH)

        correct = 0
        total = 0
        if os.path.exists(OUTPUT_SAVE_PATH):
            with open(OUTPUT_SAVE_PATH, "rb") as f:
                predictions = pickle.load(f)
            if config["task"] == "object_classification":
                eval_object_classification(dataset, predictions)
            elif config["task"] == "refCOCOg":
                eval_captioning(dataset, predictions)
            

            exit()

        model.load()
        model.eval()

        train_loader = DataLoader(dataset, 1, shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)

        responses = {}
        correct = 0
        total = 0
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
       
                output = model.generate(masks, image.copy(), questions, b_num = i)
                output = output.lower().lstrip().rstrip().replace(".", "")

                if output == answers[0]:
                    correct += 1
                total += 1
                    
                responses[i] = {}
                responses[i]["path_to_image"] = batch["path_to_image"][0]
                responses[i]["id"] = batch["id"][0]

                responses[i]["question"] = batch['question'][0]
                responses[i]["answer"] = batch['answer'][0]
                responses[i]["prediction"] = output
                #responses[i]["confidence"] = confidence.item()

                print(responses[i])
                print(correct/total)

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



    
