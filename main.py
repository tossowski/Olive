import os
import torch
from dataset.RefCOCO import RefCOCODataset
from dataset.countCOCO import CountCOCODataset
from dataset.GRIT import GRITDataset
from dataset.objectCOCO import COCOObjectDataset
from dataset.LVIS import LVISDataset
from dataset.CXR8 import CXR8Dataset
from dataset.VRP import VRPDataset
from dataset.ocr import OCRDataset
from dataset.objectInstruct import ObjectInstructDataset
from dataset.visualgenome import VisualGenomeDataset
from torch.utils.data import DataLoader
from transformers import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.olive import OLIVE

import pickle
import yaml
import argparse
import os
import random


def main(args):

    logging.set_verbosity_error()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    DATA_FOLDER = config["DATA_FOLDER"]
    config['n_patches'] = 16
    if "336" in config['vision_encoder']:
        config['n_patches'] = 24
    

    if args.train:

        print("Loading Dataset ...")
        if config['task'] == 'object_classification':
            dataset = COCOObjectDataset(config, split="train", n_patches=config['n_patches'], max_examples_per_class = config["examples_per_class"])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == 'zeroshot_object_classification':
            dataset = COCOObjectDataset(config, split="zeroshot_train", n_patches=config['n_patches'], max_examples_per_class = config["examples_per_class"])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == 'GRIT':
            dataset = GRITDataset("/data/ossowski/GRIT", n_patches=config['n_patches'])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == 'OCR':
            dataset = OCRDataset(config, n_patches=config['n_patches'])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == "open_vocab_object_classification":
            dataset = LVISDataset(config, split="val", n_patches=config['n_patches'])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2)
        elif config['task'] == 'image_captioning':
            dataset = VRPDataset("train", n_patches=config['n_patches'], use_object_annotations=config['use_object_annotations'])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == 'counting':
            dataset = CountCOCODataset(n_patches=config['n_patches'])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == 'relation_prediction':
            dataset = VisualGenomeDataset(config, split="train", n_patches=config['n_patches'])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == "refCOCOg":
            dataset = RefCOCODataset(os.path.join(DATA_FOLDER, "COCO2017", "refcocog"), os.path.join(DATA_FOLDER, "COCO2017", "train2017"), split="train", n_patches=config["n_patches"])
            train_loader = DataLoader(dataset, config["batch_size"],  shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == 'ALL':
            dataset1 = COCOObjectDataset(config, split="train", n_patches=config['n_patches'], max_examples_per_class = config["examples_per_class"])
            dataset2 = RefCOCODataset(os.path.join(DATA_FOLDER, "COCO2017", "refcocog"),  os.path.join(DATA_FOLDER, "COCO2017", "train2017"), n_patches=config["n_patches"], split="train")
            dataset3 = ObjectInstructDataset(config, n_patches=config['n_patches'])

            random.shuffle(dataset1.entries)
            random.shuffle(dataset2.entries)
            random.shuffle(dataset3.entries)
            dataset3.entries = dataset1.entries + dataset2.entries + dataset3.entries

            dataset = dataset3
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        elif config['task'] == 'ObjectInstruct':
            dataset = ObjectInstructDataset(config, n_patches=config['n_patches'])
            train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
        total = 0
        
        if config["use_retrieval"]:
            print(f"Using Retrieval with k = {config['retrieval_k']}")
            if config['task'] == "refCOCOg":
                print(f"Loading retrieval set from COCO Objects")
                retrieval_dataset = COCOObjectDataset(config, split="train", n_patches=config['n_patches'], max_examples_per_class = config["examples_per_class"])
            elif config['task'] == "ALL":
                print(f"Loading retrieval set from COCO Objects")
                retrieval_dataset = COCOObjectDataset(config, split="train", n_patches=config['n_patches'], max_examples_per_class = 1000000)
            else:
                retrieval_dataset = dataset
            model = OLIVE(config, retrieval_fn = lambda x, b: retrieval_dataset.retrieve_closest(x, config["retrieval_k"], b_num=b))    
        else:
            model = OLIVE(config, retrieval_fn = None)   

        print(f"Model SAVE/LOAD path is {model._get_save_path()}") 
        model.prepare_for_training()

        all_params = list(model.parameters())
        trainable_params = [x for x in all_params if x.requires_grad]

        optimizer = torch.optim.AdamW(trainable_params, lr=config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataset) * config['n_epochs'] // config['batch_size'], 0.000001, last_epoch=-1)

        past_losses = []
        best_avg_loss = float("inf")
        finished_training = False
        consecutive_no_improvement = 0

        start_epoch = 0
        start_batch = 0

        if args.resume:
            model.load()
            checkpoint = torch.load(os.path.join(model._get_save_path(), 'checkpoint.pth'))
            start_epoch = checkpoint['epoch']
            start_batch = checkpoint['step']
            scheduler = checkpoint['scheduler']
            optimizer.load_state_dict(checkpoint['optimizer'])

        for epoch in range(start_epoch, config["n_epochs"]):
            model.train()
            if finished_training:
                break

            for b_num, batch in enumerate(tqdm(train_loader)):
                if b_num < start_batch:
                    continue
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

                loss = out.loss
                past_losses.append(loss.item())
                loss.backward()
                optimizer.step()
                scheduler.step()
                del loss
                del out

                torch.cuda.empty_cache()
   

                if b_num > 0 and b_num % config["check_loss_steps"] == 0:
                    avg_loss = sum(past_losses[-config["check_loss_steps"]:]) / len(past_losses[-config["check_loss_steps"]:])
                    
                    if not config["early_stopping"]:
                        print(f"Loss is {round(avg_loss, 2)}")
                        SAVE_PATH = model._get_save_path()
                        os.makedirs(SAVE_PATH, exist_ok=True)

                        checkpoint = {'epoch': epoch, "step": b_num, "optimizer": optimizer.state_dict(), "scheduler": scheduler}
                        torch.save(checkpoint, os.path.join(SAVE_PATH, 'checkpoint.pth'))
                        
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
            SAVE_PATH = model._get_save_path()
            start_batch = 0
            checkpoint = {'epoch': epoch, "step": b_num, "optimizer": optimizer.state_dict(), "scheduler": scheduler}
            torch.save(checkpoint, os.path.join(SAVE_PATH, 'checkpoint.pth'))
            
            model.save()



    elif args.test:
        from PIL import Image
        from eval.utils import eval_captioning, eval_object_classification
        print("Loading Validation Dataset ...")
        if config['task'] == 'object_classification':
            dataset = COCOObjectDataset(config, split="val", n_patches=config['n_patches'], max_examples_per_class = config["examples_per_class"])
        elif config['task'] == "open_vocab_object_classification":
            dataset = LVISDataset(config, split="val", n_patches=config['n_patches'])
        elif config['task'] == 'image_captioning':
            dataset = VRPDataset("test", n_patches=config['n_patches'])
        elif config['task'] == "refCOCOg":
            dataset = RefCOCODataset(os.path.join(DATA_FOLDER, "COCO2017", "refcocog"), os.path.join(DATA_FOLDER, "COCO2017", "train2017"), n_patches=config["n_patches"], split="val")
        elif config['task'] == 'counting':
            dataset = CountCOCODataset(n_patches=config['n_patches'])
        elif config['task'] == 'medical_object_classification':
            dataset = CXR8Dataset(config, split="test", n_patches=config["n_patches"])
        elif config['task'] == 'OCR':
            dataset = OCRDataset(config, n_patches=config['n_patches'])
        elif config['task'] == 'ALL':
            #dataset = RefCOCODataset(os.path.join(DATA_FOLDER, "COCO2017", "refcocog"), os.path.join(DATA_FOLDER, "COCO2017", "train2017"), n_patches=config["n_patches"], split="val")
            dataset = COCOObjectDataset(config, split="val", n_patches=config['n_patches'], max_examples_per_class = config["examples_per_class"])
            
            #dataset1 = VRPDataset("train", n_patches=config['n_patches'], use_object_annotations=config['use_object_annotations'])
            #dataset2 = COCOObjectDataset(config, split="train", n_patches=config['n_patches'], max_examples_per_class = 1000)
            #dataset3 = OCRDataset(config, n_patches=config['n_patches'])
            #dataset3.entries.extend(dataset1.entries)
            #dataset3.entries.extend(dataset2.entries)
            #dataset = dataset3
            #train_loader = DataLoader(dataset, config["batch_size"], shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
            
        print(dataset.stats())
        if config["use_retrieval"]:
            print("Preparing Retrieval Dataset from Train Split ...")
            if config['task'] == 'object_classification':
                retrieval_dataset = COCOObjectDataset(config, split="train", n_patches=config['n_patches'], max_examples_per_class = config["examples_per_class"])
            elif config['task'] == "open_vocab_object_classification":
                retrieval_dataset = LVISDataset(config, split="train", n_patches=config['n_patches'])
            elif config['task'] == 'image_captioning':
                retrieval_dataset = VRPDataset(n_patches=config['n_patches'])
            elif config['task'] == 'counting':
                retrieval_dataset = CountCOCODataset(n_patches=config['n_patches'])
            elif config['task'] == 'medical_object_classification':
                retrieval_dataset = CXR8Dataset(config, split="train", n_patches=config["n_patches"])
            elif config['task'] == "refCOCOg":
                retrieval_dataset = COCOObjectDataset(config, split="train", n_patches=config['n_patches'], max_examples_per_class = config["examples_per_class"])
            elif config['task'] == 'ALL':
                retrieval_dataset = COCOObjectDataset(config, split="train", n_patches=config['n_patches'], max_examples_per_class = 1000000)
                
            model = OLIVE(config, retrieval_fn = lambda x, b: retrieval_dataset.retrieve_closest(x, config["retrieval_k"], train_phase=False, b_num=b))    
        else:
            model = OLIVE(config, retrieval_fn = None)    

        # Ideally make this cleaner. The load_raw = True is so that the save path includes "retrieval"
        # basically the name of the model constructed from the config, which includes the downstream task
        # However, we want to load the model from a DIFFERENT TASK. Hence we need different behavior.
        OUTPUT_SAVE_PATH = os.path.join("outputs", config["task"], config["llm_model"].split("/")[-1] + "_" + model._get_save_path(load_raw=True).split("/")[-1] + ".pkl")
        os.makedirs(os.path.join("outputs", config["task"]), exist_ok=True)
        print(OUTPUT_SAVE_PATH)

        correct = 0
        total = 0
        if os.path.exists(OUTPUT_SAVE_PATH):
            with open(OUTPUT_SAVE_PATH, "rb") as f:
                predictions = pickle.load(f)
            if config["task"] == "object_classification":
                eval_object_classification(dataset, predictions, config)
            elif config["task"] == "refCOCOg":
                eval_captioning(dataset, predictions)
            elif config["task"] == "ALL":
                
                eval_object_classification(dataset, predictions, config)
                #eval_captioning(dataset, predictions)

            #exit()

        model.load()
        model.eval()

        train_loader = DataLoader(dataset, 1, shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)

        responses = {}
        correct = 0
        exact_match = 0
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
                bbox = batch['bbox'][0]
                if config["crop_image"]:
                    img = Image.open(image[0])
                    cropped_image = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
                    output = model.generate(masks, image.copy(), questions, b_num = i, cropped_images=[cropped_image])
                else:
                    output = model.generate(masks, image.copy(), questions, b_num = i)

                output = output.lower().lstrip().rstrip().replace(".", "")

                
                if output == answers[0].lower():
                    exact_match += 1
                if answers[0].lower() in output or output in answers[0].lower():
                    correct += 1
                total += 1
                    
                responses[i] = {}
                responses[i]["path_to_image"] = batch["path_to_image"][0]
                responses[i]["id"] = batch["id"][0]
                responses[i]["question"] = batch['question'][0]
                responses[i]["answer"] = batch['answer'][0].lower()
                responses[i]["prediction"] = output

        with open(OUTPUT_SAVE_PATH, "wb") as f:
            print(f"Saved model outputs to {OUTPUT_SAVE_PATH}")
            pickle.dump(responses, f)


    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/config.yaml")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    main(args)



    
