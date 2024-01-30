import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import os
import PIL
import time
import numpy as np

from PIL import Image
from io import BytesIO
from transformers import AutoModelForCausalLM, LlamaTokenizer, CLIPImageProcessor, AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration

from models.object_encoder import ObjectEncoder
#from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model



class VisionLLaMA(nn.Module):

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def load_image(self, image_file):
        if type(image_file) == PIL.Image.Image:
            return image_file

        if image_file.startswith('http') or image_file.startswith('https'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        return image

    def __init__(self, config, only_use_shape_feature=False, retrieval_fn = None):
        super().__init__()
        self.config = config
        self.retrieval_fn = retrieval_fn

        if only_use_shape_feature:
            self.conv1 = nn.Conv2d(1, 3, kernel_size=(3, 3), stride=1, padding=1).to(config["device"])
            self.act1 = nn.ReLU().to(config["device"])
            self.drop1 = nn.Dropout(0.3).to(config["device"])
    
            
            self.pool2 = nn.MaxPool2d(kernel_size=(4, 4)).to(config["device"])
    
            self.flat = nn.Flatten().to(config["device"])
    
            self.fc3 = nn.Linear(9408, 768).to(config["device"])



        self.object_encoder = ObjectEncoder(config).to(config["device"])

        if self.config["pretrained_object_encoder_checkpoint"] != "None":
            self.object_encoder.load_state_dict(torch.load(self.config["pretrained_object_encoder_checkpoint"]))
        #self.object_encoder.requires_grad_(False)

        
        
        base_model = self.config["llm_model"]
        if "llama" in base_model:
            self.prompt_text = f"<s>[INST] <<SYS>>\n{config['system_prompt']}\n<</SYS>>\n\n"
            self.llama_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto"
        )
        elif "gpt2" in base_model:
            self.prompt_text = ""
            self.llama_model = AutoModelForCausalLM.from_pretrained(
            base_model
        ).to(self.config["device"])
        elif "llava" in base_model:
            self.prompt_text = "<image>\nUSER: "
            self.llama_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", device_map="auto")
        
        

        #print(self.llama_model)

        if "llama" in base_model.lower():
            default_layers = len(self.llama_model.model.layers)
        elif "gpt2" in base_model:
            default_layers = len(self.llama_model.transformer.h)
        elif "llava" in base_model:
            default_layers = 0 # Don't resize llava
        
        if self.config["n_decoder_layers"] < default_layers:
            n_layers = self.config["n_decoder_layers"]
            print(f"Resizing decoder layers to {n_layers} ...")
            
            if "llama" in base_model.lower():
                self.llama_model.model.layers = self.llama_model.model.layers[:self.config["n_decoder_layers"]]
            elif "gpt2" in base_model:
                self.llama_model.transformer.h = self.llama_model.transformer.h[:self.config["n_decoder_layers"]]
            #print(self.llama_model)

        if "llama" in base_model.lower():
            self.tokenizer = LlamaTokenizer.from_pretrained(base_model)
        elif "gpt2" in base_model:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        elif "llava" in base_model.lower():
            processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            self.tokenizer = processor.tokenizer
            self.image_processor = processor.image_processor

        if "llava" in self.config["llm_model"]:
            self.object_encoder.projector  = self.llama_model.multi_modal_projector
            self.object_encoder.processor = self.image_processor
            self.object_encoder.model = self.llama_model.vision_tower
            self.object_encoder.projector.requires_grad_(False)
            assert self.config["patch_size"] == 24

        self.tokenizer.padding_side = "right"
        if "llava" in base_model:
            self.tokenizer.padding_size = "left"
        
        self.tokenizer.add_tokens(["[obj]"])
        self.tokenizer.add_tokens(["[start]"])

        if "llama" in base_model or "llava" in base_model:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            #self.tokenizer.pad_token = self.tokenizer.eos_token
        elif "gpt2" in base_model:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


        self.llama_model.resize_token_embeddings(len(self.tokenizer))

        if "llava" in base_model:
            self.decode_start_token = self.tokenizer.convert_tokens_to_ids(":")
        else:
            self.decode_start_token = self.tokenizer.convert_tokens_to_ids("[start]")

        # self.prompt = self.tokenizer(self.prompt_text).input_ids


        self.obj_token_id = self.tokenizer.convert_tokens_to_ids("[obj]")
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids("[PAD]")

        if self.config["freeze_llm"]:
            self.llama_model.requires_grad_(False)
            # if "gpt2" in base_model:
            #     self.llama_model.lm_head.requires_grad_(True)

        #print(self.llama_model)
            #self.llama_model.lm_head.requires_grad_(True)
        #print(sum(p.numel() for p in self.llama_model.parameters() if p.requires_grad))
        

    def prepare_for_training(self):
        if not self.config["freeze_llm"]:
            peft_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.1,
                r=64,
                bias="none",
                task_type="CAUSAL_LM", 
            )

            self.llama_model = prepare_model_for_kbit_training(self.llama_model)
            self.llama_model = get_peft_model(self.llama_model, peft_config)

    def resize_tensor(self, input_tensor, new_size):
        # Assuming input_tensor is of shape (batch_size, channels, height, width)

        # Perform the resizing
        resized_tensor = F.interpolate(input_tensor, size=new_size, mode='bilinear', align_corners=False)

        return resized_tensor

    # special_values: list of elements. each element is (n_segmentations x 4096)
    def embed_with_special_tokens(self, sentences, special_values, image_features=None, labels=None):
        tokenizer_output = self.tokenizer(sentences, padding=True)
        attention_mask = torch.tensor(tokenizer_output.attention_mask, dtype=torch.long).to(self.config["device"])
        batch_tokens = torch.tensor(tokenizer_output.input_ids, dtype=torch.long).to(self.config["device"])

        if "llava" in self.config["llm_model"]:
            input_ids = torch.tensor(self.tokenizer(sentences, padding=True).input_ids, dtype=torch.long).to(self.config["device"])
            #pixel_values = self.image_processor([image[0] for image in images], return_tensors='pt')['pixel_values']
            inputs_embeds = self.llama_model.get_input_embeddings()(input_ids).to(self.config["device"])

            
            inputs_embeds, attention_mask, batch_tokens, _ = self.llama_model._merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids, attention_mask, input_ids
            )

            batch_tokens[:, 1:578] = 1 # We will override this with the image features layer. Otherwise this causes an embedding lookup error because it is filled with -100
            # Just need to make sure to reset it later

            inputs_embeds = inputs_embeds.to(self.config["device"])
            batch_tokens = batch_tokens.to(self.config["device"])

            if labels is not None:
                labels = batch_tokens

            final_image_input = inputs_embeds[:, 1:578, :]

        

        embed_layer = self.llama_model.get_input_embeddings().to(self.config["device"])

        if len(special_values) == 0:
            if "llava" in self.config["llm_model"]:
                batch_tokens[:, 1:578] = -100
                return inputs_embeds, batch_tokens, attention_mask.to(self.config["device"])
            input_embeds = embed_layer(batch_tokens).to(self.config["device"])
            
            return input_embeds, input_embeds, attention_mask.to(self.config["device"])
       
        new_embeds = []
        new_labels = [] if labels is not None else None
        new_attn_mask = []

        for i, sent in enumerate(batch_tokens):
            cur_new_embeds = []
            cur_new_attn_mask = []
            cur_obj_index = 0
            cur_obj_features = special_values[i]
            cur_attn_mask = attention_mask[i]
            obj_token_indices = torch.where(sent == self.obj_token_id)[0]

            
            if labels is not None:
                cur_labels = labels[i]
                cur_new_labels = []
                assert cur_labels.shape == sent.shape

            while obj_token_indices.numel() > 0:
                cur_object_features = cur_obj_features[cur_obj_index : cur_obj_index + 1, :]
                obj_token_index = obj_token_indices[0].item()
     
                if obj_token_index > 0:
                    cur_new_embeds.append(embed_layer(sent[:obj_token_index]))

                cur_new_embeds.append(cur_object_features)
                
                if labels is not None:

                    
                    cur_new_labels.append(cur_labels[:obj_token_index])
                    cur_new_labels.append(torch.full((cur_object_features.shape[0],), -100, device=labels.device, dtype=labels.dtype))
                    cur_labels = cur_labels[obj_token_index + 1:]

                cur_new_attn_mask.append(cur_attn_mask[:obj_token_index])
                cur_new_attn_mask.append(torch.full((cur_object_features.shape[0],), 1, device=attention_mask.device, dtype=attention_mask.dtype))
                cur_attn_mask = cur_attn_mask[obj_token_index + 1:]

                sent = sent[obj_token_index+1:]
                obj_token_indices = torch.where(sent == self.obj_token_id)[0]
                cur_obj_index += 1


            if sent.numel() > 0:
                cur_new_embeds.append(embed_layer(sent))
                cur_new_attn_mask.append(cur_attn_mask)
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            

            #cur_new_embeds = [x.to(device=self.config["device"]) for x in cur_new_embeds]
            
            # Combining object + word features in a sentence

            cur_new_embeds = torch.cat(cur_new_embeds, dim=0)

            
            cur_new_attn_mask = torch.cat(cur_new_attn_mask, dim=0)


            new_attn_mask.append(cur_new_attn_mask)
            new_embeds.append(cur_new_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)

                if "llava" in self.config["llm_model"]:
                    cur_new_labels[1:578] = -100
                new_labels.append(cur_new_labels)
            

            

        #print([x.shape for x in new_embeds])
        # Combining embeds for each sentence to get batch embedding
        #print([x.shape for x in new_embeds])
        
        new_input_embeds = torch.stack(new_embeds, dim=0)
        new_attn_mask = torch.stack(new_attn_mask, dim=0)
        if labels is not None:
            new_labels  = torch.stack(new_labels, dim=0).to(self.config["device"])

            

        #return torch.cat(new_embeds, axis = 0)
        #print(new_labels[0])
        if "llava" in self.config["llm_model"]:
            new_input_embeds[:, 1:578, :] = final_image_input
        #print(new_input_embeds)
        #print(new_labels)
        return new_input_embeds.to(self.config["device"]), new_labels, new_attn_mask.to(self.config["device"])

    def prepare_input(self, segmentations, images, sentences, labels=None, return_retrieved_info = False):
 
        if self.config["use_retrieval"]:

            prompts, retrieved_masks, retrieved_images = self.get_retrieval_prompt(segmentations, images)
            new_segs = [segmentations[i, :] for i in range(len(segmentations))]

            for i in range(len(prompts)):
                if "[obj]" not in sentences[i]:
                    continue
                sentences[i] = prompts[i] + sentences[i]

                val = segmentations[i]
                if len(segmentations[i].shape) == 1:
                    val = torch.unsqueeze(segmentations[i], 0).to(self.config["device"])
                #print(segmentations[i].shape, retrieved_masks[i].shape)
                new_segs[i] = torch.cat((retrieved_masks[i], val), axis = 0)

                if type(images[i]) == list:
                    images[i] = retrieved_images[i] + images[i]
                else:
                    images[i] = retrieved_images[i] + [images[i]]
            segmentations = new_segs
        for i in range(len(images)):
            if type(images[i]) == list:
                images[i] = [self.load_image(x) for x in images[i]]
            else:
                images[i] = [self.load_image(images[i])]


        # for i in range(len(segmentations)):
        #     segmentations[i] = torch.BoolTensor(segmentations[i]).to(self.config["device"])
           
        if labels == None:
            if "llama" in self.config["llm_model"]:
                labels = [" [/INST] [start]" for sent in sentences]
            elif "gpt2" in self.config["llm_model"]:
                labels = [" [start] " for sent in sentences]
            elif "llava" in self.config["llm_model"]:
                labels = ["\nASSISTANT: " for _ in sentences]
        else:
            if "llama" in self.config["llm_model"]:
                sentences = [sent + " [/INST] " for sent in sentences]
                labels = ["[start] " + label + "</s>" for label in labels]
            elif "gpt2" in self.config["llm_model"]:
                labels = [" [start] " + label + "." for label in labels]
            elif "llava" in self.config["llm_model"]:
                labels = ["\nASSISTANT: " + label for label in labels]

        full_text_input = [self.prompt_text + sentences[i] + labels[i] for i in range(len(sentences))]
        object_embeddings = []
        image_features = []

        if len(segmentations) > 0:
            for i in range(len(images)):
                inputs = self.object_encoder.processor(images=images[i], return_tensors="pt").to(self.config["device"])

                transformer_output = self.object_encoder.model(**inputs).last_hidden_state
                image_features.append(transformer_output)

                mask_input = segmentations[i]
                if len(mask_input.shape) == 1:
                    mask_input = torch.unsqueeze(mask_input, 0)

                # If user query involves multiple objects, duplicate image feature
                if transformer_output.shape[0] != mask_input.shape[0]:
                    transformer_output = transformer_output.repeat(mask_input.shape[0], 1, 1)

                object_embedding = self.object_encoder(mask_input, transformer_output)
                object_embeddings.append(object_embedding)
        # else:
            #print(full_segmentations)
            # full_segmentations = [self.resize_tensor(x.unsqueeze(0).unsqueeze(0), (224,224)) for x in full_segmentations]
            # for seg in full_segmentations:
            #     x = self.act1(self.conv1(seg))
            #     x = self.drop1(x)
            #     x = self.pool2(x)
            #     x = self.flat(x)
            #     object_embeddings.append(self.fc3(x))
            #object_embeddings = torch.stack(object_embeddings, axis = 0)
           # print(object_embeddings.shape)
            


        labels = torch.tensor(self.tokenizer(full_text_input, padding=True).input_ids, dtype=torch.long).to(self.config["device"])
        if "llava" in self.config["llm_model"]:
            #print(len(image_features))
            #print(image_features[0].shape)
            image_features = torch.cat(image_features, 0).to(self.config["device"])
            image_features = self.object_encoder.projector(image_features)
            print(image_features.shape)
            final_input, label_input, attention_mask = self.embed_with_special_tokens(full_text_input, object_embeddings, image_features=image_features, labels=labels)
        else:
            final_input, label_input, attention_mask = self.embed_with_special_tokens(full_text_input, object_embeddings, labels=labels)
        
        if return_retrieved_info:
            return final_input, label_input, attention_mask, prompts, retrieved_masks, retrieved_images
        return final_input, label_input, attention_mask

    def get_retrieval_prompt(self, mask, images):
       
        inputs = self.object_encoder.processor(images=[self.load_image(x) for x in images], return_tensors="pt").to(self.config["device"])
            
        image_forward_outs = self.object_encoder.model(inputs['pixel_values'], output_hidden_states=True)

        if type(mask) == list:
            mask = torch.stack(mask, axis = 0)[:, 1:]
        else:
            mask = mask[:, 1:]
        object_features = []
        for i in range(len(mask)):
            image_feat = image_forward_outs.hidden_states[-1][i,1:,:][mask[i]].detach().cpu()
            object_feature = torch.mean(image_feat, dim = 0)
            object_feature /= object_feature.norm(dim=-1, keepdim=True)
            object_features.append(object_feature)
        object_features = torch.stack(object_features, axis = 0)
        closest_entries, similarity_scores = self.retrieval_fn(object_features)
        prompts = []
        masks = []
        all_images = []
        for i in range(len(closest_entries)):
            prompt = f"The top {self.config['retrieval_k']} related objects are:\n"
            entries = closest_entries[i]
            images = []
            segmentations = []
            for x, entry in enumerate(entries):
                prompt += f"[obj] {entry['answer']} with confidence {similarity_scores[i][x]}\n"
                segmentations.append(torch.BoolTensor(entry["vit_mask"]).to(self.config["device"]))
                images.append(entry["path_to_image"])
            prompt += "\n"
            prompts.append(prompt)
            #print(torch.stack(segmentations, axis = 0).shape)
            masks.append(torch.stack(segmentations, axis = 0))
            all_images.append(images)
        #print(torch.stack(masks, axis = 0).shape)
    
        return prompts, masks, all_images

        #print(closest_entries)

    # Sentences: list of strings (questions)
    # Labels: List of strings (the answers)
    # Image: list of list of paths to images
    # Segmentations: List of list of Binary 16x16 mask
    def forward(self, segmentations, images, sentences, labels=None):
        
            #print(images)
        final_input, label_input, attention_mask = self.prepare_input(segmentations, images, sentences, labels)
        
        for i in range(len(label_input)):
            idx = (label_input[i] == self.decode_start_token).nonzero(as_tuple=True)[0][-1]
            #print((label_input[i] == self.decode_start_token).nonzero(as_tuple=True))

            #print(idx)
            label_input[i, :idx + 1] = -100
        #print(label_input)
    
        return self.llama_model(inputs_embeds = final_input, labels = label_input, attention_mask=attention_mask)

    

    def generate(self, segmentations, images, sentences, return_retrieved_info=False):

        if return_retrieved_info:
            final_input, label_input, attention_mask, prompts, masks, images = self.prepare_input(segmentations, images, sentences, labels=None, return_retrieved_info=True)
        else:
            final_input, label_input, attention_mask = self.prepare_input(segmentations, images, sentences, labels=None)
        
        out = self.llama_model.generate(inputs_embeds = final_input, attention_mask = attention_mask, max_new_tokens=30, top_p=0.0, top_k=1)
        #out = self.llama_model.generate(inputs_embeds = final_input, attention_mask = attention_mask, max_new_tokens=30)

 
        decoded_output = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        #print(decoded_output)
        if len(decoded_output) == 1:
            decoded_output = decoded_output[0]

        if return_retrieved_info:

            
            return decoded_output, prompts, masks, images

        return decoded_output

    def _get_save_path(self):
        SAVE_PATH = ""
        if self.config["freeze_llm"]:
            SAVE_PATH += "frozen_llm_"
        else:
            SAVE_PATH += "finetuned_llm_"

        if self.config["pretrained_object_encoder_checkpoint"] != "None":
            SAVE_PATH += "obj_encoder_checkpoint_"

        if "dino" in self.config["vision_encoder"]:
            SAVE_PATH += "dino_"
        elif "clip" in self.config["vision_encoder"]:
            SAVE_PATH += "clip_"

        if self.config["early_stopping"]:
            SAVE_PATH += "early_stopping_"

        if self.config["n_decoder_layers"] < 32:
            n_layers = self.config["n_decoder_layers"]
            SAVE_PATH += f"{n_layers}_decoder_layers_"

        if self.config["use_only_shape_feature"]:
            SAVE_PATH += "shape_feature_only_"

        if self.config["use_retrieval"]:
            SAVE_PATH += "retrieval_"
        
        # if not self.config["use_CLS_emb"]:
        #     SAVE_PATH += "no_CLS_emb"

        if self.config["task"] == "image_captioning":
            if self.config["use_object_annotations"]:
                SAVE_PATH += "object_level_annotations_"
            

        

        patch_size = self.config["patch_size"]
        SAVE_PATH += f"{patch_size}x{patch_size}_patches_"

        #SAVE_PATH += f"{self.config['examples_per_class']}_examples_per_class_"
        
        SAVE_PATH = SAVE_PATH[:-1]
        if "llama" in self.config["llm_model"].lower():
            folder = "llama_2_finetuned_checkpoints"
        elif "llava" in self.config['llm_model'].lower():
            folder = "llava_finetuned_checkpoints"
        elif "gpt" in self.config["llm_model"].lower():
            folder = "gpt2_finetuned_checkpoints"
        SAVE_PATH = os.path.join(self.config["save_folder"], folder, self.config["task"], SAVE_PATH)
        return SAVE_PATH

    def load(self):
        SAVE_PATH = self._get_save_path()
        print(SAVE_PATH)
        self.object_encoder.load_state_dict(torch.load(os.path.join(SAVE_PATH, "llama_2_7b_adapter_finetuned")))
        print(f"Loaded LLM model and Object Encoder from {SAVE_PATH}")
        if self.config["freeze_llm"]:
            if "gpt2" in self.config["llm_model"]:
                return
                self.llama_model = AutoModelForCausalLM.from_pretrained(
                    SAVE_PATH).to(self.config["device"])
            return
        
        self.llama_model = PeftModel.from_pretrained(self.llama_model, SAVE_PATH)
        
        

    def save(self):
        

        SAVE_PATH = self._get_save_path()
        os.makedirs(SAVE_PATH, exist_ok=True)
        # if self.config[""]

        #os.makedirs(os.path.join(self.config["save_path"], "llama_2_7b_adapter_finetuned"), exist_ok=True)
        if not self.config["freeze_llm"] or "gpt2" in self.config["llm_model"]:
            self.llama_model.save_pretrained(SAVE_PATH)
        
        torch.save(self.object_encoder.state_dict(), os.path.join(SAVE_PATH, "llama_2_7b_adapter_finetuned"))
        print(f"Saved model to {SAVE_PATH}")