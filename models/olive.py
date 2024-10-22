import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import os
import PIL
import numpy as np

from PIL import Image
from io import BytesIO
from transformers import AutoModelForCausalLM, LlamaTokenizer, CLIPImageProcessor, CLIPVisionModel, AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration, AutoImageProcessor, AutoModel
from collections import Counter
from models.object_encoder import ObjectEncoder
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model


class OLIVE(nn.Module):

    # Counts the number of trainable parameters in the model
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # Helper function to load an image given the path to the image.
    def load_image(self, image_file):
        if type(image_file) == PIL.Image.Image:
            return image_file

        if image_file.startswith('http') or image_file.startswith('https'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        return image

    # Retrieval function: a function which takes in query object features and returns
    # information about retrieved objects. See the dataset classes for implementation
    def __init__(self, config, retrieval_fn = None):
        super().__init__()
        self.config = config
        self.retrieval_fn = retrieval_fn
        self.object_encoder = ObjectEncoder(config).to(config["device"])

        base_model = self.config["llm_model"]

        # Set up settings based on backbone LLM
        if "llama" in base_model:
            self.prompt_text = f"<s>[INST] <<SYS>>\n{config['system_prompt']}\n<</SYS>>\n\n"
            self.llama_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto"
        )
            self.tokenizer = LlamaTokenizer.from_pretrained(base_model)
            self.tokenizer.padding_side = "right"
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        elif "gpt2" in base_model:
            self.prompt_text = ""
            self.llama_model = AutoModelForCausalLM.from_pretrained(
            base_model
        ).to(self.config["device"])
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.padding_side = "right"
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        elif "llava" in base_model:
            self.prompt_text = "<image>\nUSER: "
            self.llama_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", device_map="auto")
            processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            self.tokenizer = processor.tokenizer
            self.image_processor = processor.image_processor
            self.object_encoder.projector = self.llama_model.multi_modal_projector
            self.object_encoder.processor = self.image_processor
            self.object_encoder.model = self.llama_model.vision_tower
            self.object_encoder.model.requires_grad_(False)
            self.object_encoder.projector.requires_grad_(False)
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.padding_side = "right"
            assert self.config["n_patches"] == 24
        
        # Add special token for object. Gets replaced with obj vector during
        # tokenization in embed_with_special_tokens
        self.tokenizer.add_tokens(["[obj]"])

        # This token is a dummy to help with finding the start of generation
        # We mask everything with -100 before this token so it doesn't count
        # towards the loss
        self.tokenizer.add_tokens(["[start]"])
        
        self.llama_model.resize_token_embeddings(len(self.tokenizer))

        if "llava" in base_model:
            self.decode_start_token = self.tokenizer.convert_tokens_to_ids(":")
        else:
            self.decode_start_token = self.tokenizer.convert_tokens_to_ids("[start]")

        self.obj_token_id = self.tokenizer.convert_tokens_to_ids("[obj]")
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids("[PAD]")

        print(f"Initialized model with {self.config['llm_model']} LLM backbone and {self.config['vision_encoder']} Vision Encoder")
        

        if self.config["freeze_llm"]:
            print("The LLM is FROZEN")
            self.llama_model.requires_grad_(False)


    def prepare_for_training(self):
        if not self.config["freeze_llm"] and "gpt2" not in self.config["llm_model"]:
            peft_config = LoraConfig(
                lora_alpha=256,
                lora_dropout=0.1,
                r=128,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head"]
            )

            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
        else:
            print(f"There are {self.count_trainable_parameters()} trainable parameters")

    # Key function which replaces [obj] tokens in input with their vector representation
    # special_values: list of tensor of object vectors. each element is (n_segmentations x 4096)
    # sentences: list of user text inputs
    # image_features: If supplied, will concatenate the patch image features to the beginning of input embeds
    # labels: A copy of sentences
    def embed_with_special_tokens(self, sentences, special_values, image_features=None, labels=None):
        tokenizer_output = self.tokenizer(sentences, padding=True)
        attention_mask = torch.tensor(tokenizer_output.attention_mask, dtype=torch.long).to(self.config["device"])
        batch_tokens = torch.tensor(tokenizer_output.input_ids, dtype=torch.long).to(self.config["device"])

        if len(image_features) > 0:
            image_features = torch.cat(image_features, dim = 0).to(self.config["device"])
            input_ids = torch.tensor(self.tokenizer(sentences, padding=True).input_ids, dtype=torch.long).to(self.config["device"])
            #pixel_values = self.image_processor([image[0] for image in images], return_tensors='pt')['pixel_values']
            inputs_embeds = self.llama_model.get_input_embeddings()(input_ids).to(self.config["device"])

            
            inputs_embeds, attention_mask, batch_tokens, _ = self.llama_model._merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids, attention_mask, input_ids
            )

            batch_tokens[:, 1:(self.config["n_patches"] ** 2) + 2] = 1 # We will override this with the image features later. Otherwise this causes an embedding lookup error because it is filled with -100
            # Just need to make sure to reset it later

            inputs_embeds = inputs_embeds.to(self.config["device"])
            batch_tokens = batch_tokens.to(self.config["device"])

            if labels is not None:
                labels = batch_tokens

            final_image_input = inputs_embeds[:, 1:(self.config["n_patches"] ** 2) + 2, :]


        embed_layer = self.llama_model.get_input_embeddings().to(self.config["device"])

        if len(special_values) == 0:
            if self.config["use_image_features"]:
                batch_tokens[:, 1:(self.config["n_patches"] ** 2) + 2] = -100
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

            cur_new_embeds = torch.cat(cur_new_embeds, dim=0)

            
            cur_new_attn_mask = torch.cat(cur_new_attn_mask, dim=0)


            new_attn_mask.append(cur_new_attn_mask)
            new_embeds.append(cur_new_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)

                if self.config["use_image_features"]:
                    cur_new_labels[1:(self.config["n_patches"] ** 2) + 2] = -100
                new_labels.append(cur_new_labels)
        
        new_input_embeds = torch.stack(new_embeds, dim=0)
        new_attn_mask = torch.stack(new_attn_mask, dim=0)
        if labels is not None:
            new_labels  = torch.stack(new_labels, dim=0).to(self.config["device"])

        if self.config["use_image_features"]:
            new_input_embeds[:, 1:(self.config["n_patches"] ** 2) + 2, :] = final_image_input
 
        # if len(image_features) > 0:
        #     image_features = torch.cat(image_features, axis = 0).to(self.config["device"])
        #     n_patches = 1 + self.config["n_patches"] ** 2
        #     new_input_embeds = torch.cat((image_features, new_input_embeds), axis = 1)
        #     new_labels = torch.cat((-100 * torch.ones((self.config["batch_size"], n_patches), dtype=torch.long).to(self.config["device"]), new_labels), axis = 1)
        #     new_attn_mask = torch.cat((torch.ones((self.config["batch_size"], n_patches), dtype=torch.long).to(self.config["device"]), new_attn_mask), axis = 1)
        
        #print(new_input_embeds.shape, new_labels.shape, new_attn_mask.shape)
        return new_input_embeds.to(self.config["device"]), new_labels, new_attn_mask.to(self.config["device"])

    def prepare_input(self, segmentations, images, sentences, labels=None, return_retrieved_info = False, b_num=0, cropped_images=[]):

        # Augment prompt with retrieved object vectors and images
        if self.config["use_retrieval"] and len(segmentations) > 0:
            if self.config["crop_image"]:
                prompts, retrieved_masks, retrieved_images = self.get_retrieval_prompt(segmentations, images, b_num=b_num, cropped_images=cropped_images)
            else:
                prompts, retrieved_masks, retrieved_images = self.get_retrieval_prompt(segmentations, images, b_num=b_num)
            new_segs = [segmentations[i] for i in range(len(segmentations))]

            for i in range(len(prompts)):
                if "[obj]" not in sentences[i]:
                    continue
                sentences[i] = prompts[i] + sentences[i]

                val = segmentations[i]
                if len(segmentations[i].shape) == 1:
                    val = torch.unsqueeze(segmentations[i], 0).to(self.config["device"])
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

        
        if labels == None:
            if "llama" in self.config["llm_model"]:
                labels = [" [/INST] [start]" for sent in sentences]
            elif "gpt2" in self.config["llm_model"]:
                labels = [" [start]" for sent in sentences]
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
   
        # Get object features based on the masks
        if len(segmentations) > 0:
            for i in range(len(segmentations)):
                inputs = self.object_encoder.processor(images=images[i], return_tensors="pt").to(self.config["device"])
                transformer_output = self.object_encoder.model(**inputs).last_hidden_state
                if self.config["use_image_features"]:
                    if "llava" in self.config["llm_model"]:
                        image_features.append(self.llama_model.multi_modal_projector(transformer_output))


                mask_input = segmentations[i]
                if len(mask_input.shape) == 1:
                    mask_input = torch.unsqueeze(mask_input, 0)

                # If user query involves multiple objects, duplicate image feature
                if transformer_output.shape[0] != mask_input.shape[0]:
                    transformer_output = transformer_output.repeat(mask_input.shape[0], 1, 1)

                object_embedding = self.object_encoder(mask_input, transformer_output)
                object_embeddings.append(object_embedding)
        elif self.config["use_image_features"] and len(images) > 0:
            inputs = self.object_encoder.processor(images=images[i], return_tensors="pt").to(self.config["device"])
            transformer_output = self.object_encoder.model(**inputs).last_hidden_state
            image_features.append(self.llama_model.multi_modal_projector(transformer_output))
  
        labels = torch.tensor(self.tokenizer(full_text_input, padding=True).input_ids, dtype=torch.long).to(self.config["device"])
        
        # Replace obj tokens with their object vector representation
        final_input, label_input, attention_mask = self.embed_with_special_tokens(full_text_input, object_embeddings, labels=labels, image_features=image_features)
        
        if return_retrieved_info:
            return final_input, label_input, attention_mask, prompts, retrieved_masks, retrieved_images
        return final_input, label_input, attention_mask

    # Returns the
    # prompts: Few shot raw text prompts
    # masks: List of ViT object binary mask of shape (retrieval_k, patch_size ** 2 + 1)
    # all_images: List of list of images. Each inner list has length retrieval_k
    def get_retrieval_prompt(self, mask, images, b_num = 0, cropped_images = []):
        object_features = []

        if len(cropped_images) > 0:
            inputs = self.object_encoder.processor(images=[self.load_image(x[0]) if type(x) == list else self.load_image(x) for x in images], return_tensors="pt").to(self.config["device"])
            image_forward_outs = self.object_encoder.model(inputs['pixel_values'], output_hidden_states=True)
            object_feature = image_forward_outs.hidden_states[-1][0,0,:]
            object_feature /= object_feature.norm(dim=-1, keepdim=True)
            object_features.append(object_feature)
        else:

            # For input questions with multiple objects (type(x) == list), for now assume from the same image
            inputs = self.object_encoder.processor(images=[self.load_image(x[0]) if type(x) == list else self.load_image(x) for x in images], return_tensors="pt").to(self.config["device"])
            
            image_forward_outs = self.object_encoder.model(inputs['pixel_values'], output_hidden_states=True)

            if type(mask) == list:
                mask_copy = mask.copy()
                for i, m in enumerate(mask_copy):
                    if len(m.shape) > 1:
                        mask_copy[i] = m[0]
                mask_copy = torch.stack(mask_copy, axis = 0)[:, 1:]
            else:
                mask_copy = mask[:, 1:]

            object_features = []
            for i in range(len(mask_copy)):
                image_feat = image_forward_outs.hidden_states[-1][i,1:,:][mask_copy[i]]
                object_feature = torch.mean(image_feat, dim = 0)
                object_feature /= object_feature.norm(dim=-1, keepdim=True)
                object_features.append(object_feature)
                
        object_features = torch.stack(object_features, axis = 0).to(self.config["device"])
        closest_entries, similarity_scores = self.retrieval_fn(object_features, b_num)
        prompts = []
        masks = []
        all_images = []

        for i in range(len(closest_entries)):
            entries = closest_entries[i]
            answers = [entry['answer'] for entry in entries]
            majority_element = max(Counter(answers), key=Counter(answers).get)
            
            if self.config['majority_vote_retrieval']:
                prompt = f"The top {Counter(answers).get(majority_element)} related objects are:\n"

            else:
                prompt = f"The top {self.config['retrieval_k']} related objects are:\n"
            images = []
            segmentations = []
            for x, entry in enumerate(entries):
                if self.config['majority_vote_retrieval'] and entry["answer"] != majority_element:
                    continue
                prompt += f"[obj] {entry['answer']} with confidence {similarity_scores[i][x]:.2}\n"
                segmentations.append(torch.BoolTensor(entry["vit_mask"]).to(self.config["device"]))
                images.append(entry["path_to_image"])
            prompt += "\n"
            prompts.append(prompt)
            masks.append(torch.stack(segmentations, axis = 0))
            all_images.append(images)


        return prompts, masks, all_images

    # Sentences: list of strings (questions)
    # Labels: List of strings (the answers)
    # Image: list of list of paths to images
    # Segmentations: List of (n_segmentations, patch_size ** 2 + 1)-shape tensors.
    # A training example may have multiple object vectors in the prompt
    def forward(self, segmentations, images, sentences, labels=None, output_hidden_states=False):
        final_input, label_input, attention_mask = self.prepare_input(segmentations, images, sentences, labels)
        
        # Mask up to start decoding token so it doesn't count to loss
        for i in range(len(label_input)):
            idx = (label_input[i] == self.decode_start_token).nonzero(as_tuple=True)[0][-1]
            label_input[i, :idx + 1] = -100

        return self.llama_model(inputs_embeds = final_input, labels = label_input, attention_mask=attention_mask, output_hidden_states=output_hidden_states)

    

    def generate(self, segmentations, images, sentences, return_retrieved_info=False, b_num = 0, cropped_images = []):
        if return_retrieved_info:
            final_input, label_input, attention_mask, prompts, masks, images = self.prepare_input(segmentations, images, sentences, labels=None, return_retrieved_info=True, b_num=b_num, cropped_images = cropped_images)
        else:
            final_input, label_input, attention_mask = self.prepare_input(segmentations, images, sentences, labels=None, b_num=b_num, cropped_images = cropped_images)
        
        out = self.llama_model.generate(inputs_embeds = final_input, attention_mask = attention_mask, max_new_tokens=100, top_p=0.0, top_k=1)
        decoded_output = self.tokenizer.batch_decode(out, skip_special_tokens=True)

        if len(decoded_output) == 1:
            decoded_output = decoded_output[0]

        if return_retrieved_info:

            
            return decoded_output, prompts, masks, images

        return decoded_output

    def _get_save_path(self, load_raw=False):

        if "load_model_path" in self.config and not load_raw:
            return self.config["load_model_path"]

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
            if "336" in self.config["vision_encoder"]:
                SAVE_PATH += "clip_336_"
            else:
                SAVE_PATH += "clip_"

        if self.config["early_stopping"]:
            SAVE_PATH += "early_stopping_"

        if self.config["n_decoder_layers"] < 32:
            n_layers = self.config["n_decoder_layers"]
            SAVE_PATH += f"{n_layers}_decoder_layers_"

        if self.config["use_retrieval"]:
            SAVE_PATH += "retrieval_"        

        if self.config["no_compression"]:
            SAVE_PATH += "no_compression_"       

        if self.config["task"] == "image_captioning":
            if self.config["use_object_annotations"]:
                SAVE_PATH += "object_level_annotations_"
            

        

        n_patches = self.config["n_patches"]
        SAVE_PATH += f"{n_patches}x{n_patches}_patches_"
        
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
        print(f"The save path is: {SAVE_PATH}")
        self.object_encoder.load_state_dict(torch.load(os.path.join(SAVE_PATH, "llama_2_7b_adapter_finetuned")))
        print(f"Loaded Object Encoder checkpoint from {SAVE_PATH}")
        if self.config["freeze_llm"]:
            return
        if "gpt2" in self.config["llm_model"]:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                SAVE_PATH).to(self.config["device"])
            return
        
        self.llama_model = PeftModel.from_pretrained(self.llama_model, SAVE_PATH)
        

    def save(self):

        SAVE_PATH = self._get_save_path()
        os.makedirs(SAVE_PATH, exist_ok=True)

        if not self.config["freeze_llm"] or "gpt2" in self.config["llm_model"]:
            self.llama_model.save_pretrained(SAVE_PATH)
        
        torch.save(self.object_encoder.state_dict(), os.path.join(SAVE_PATH, "llama_2_7b_adapter_finetuned"))
        print(f"Saved model to {SAVE_PATH}")