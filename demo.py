import yaml
from models.olive import OLIVE
from dataset.objectCOCO import COCOObjectDataset
import gradio as gr
import time
import numpy as np
import matplotlib.pyplot as plt
import skimage
import math
import torch
from PIL import Image
#from dataset.CXR8 import CXR8Dataset

with open("configs/config.yaml", 'r') as file:
        config = yaml.safe_load(file)

config['n_patches'] = 16
if "336" in config['vision_encoder']:
        config['n_patches'] = 24
        
if config["use_retrieval"]:
        #dataset = CXR8Dataset(config, split="train", n_patches=config["n_patches"])
        dataset = COCOObjectDataset(config, split="train", n_patches=config['n_patches'], max_examples_per_class = 1000000)

model = None
old_config = None


def _get_ViT_mask(mask, height, width, output_height, output_width):
    pooled_mask = skimage.measure.block_reduce(mask, block_size=(math.floor(height / output_height), math.floor(width / output_width)), func=np.max)

    result_height, result_width = pooled_mask.shape
    # If the result is smaller than 16x16, pad it with zeros
    if result_height < output_height or result_width < output_width:
        pad_height = output_height - result_height
        pad_width = output_width - result_width
        pooled_mask = np.pad(pooled_mask, ((0, pad_height), (0, pad_width)), mode='constant')

    if result_height > output_height or result_width > output_width:
        pooled_mask = pooled_mask[:output_height, :output_width]

    assert pooled_mask.shape == (output_height,output_width)
    return torch.BoolTensor(np.append(1, pooled_mask.flatten()))

def sleep(im):
    time.sleep(2)
    ret = [im["background"]]
    for layer in im["layers"]:
        ret.append(layer)
    return ret

def generate_predictions(question, images, task, backbone, use_retrieval, freeze_llm, chat_history):
    global model
    global old_config
    image = Image.open(images.root[0].image.path).convert('RGB')
    segmentations = [Image.open(x.image.path).convert('RGB') for x in images.root[1:]]
    
    config['freeze_llm'] = freeze_llm
    config['llm_model'] = backbone
    config['task'] = task
    config['use_retrieval'] = use_retrieval

    if "llama" or "gpt2" in backbone:
        if "336" in config["vision_encoder"]:
            output_width, output_height = 24, 24
        else:
            output_width, output_height = 16, 16
        
    elif "llava" in backbone:
        output_width, output_height = 24, 24

    config['n_patches'] = output_width

    if old_config != config:
        if config['use_retrieval']:
            model = OLIVE(config, retrieval_fn = lambda x, y: dataset.retrieve_closest(x, config["retrieval_k"], train_phase=False, b_num = y))    
        else:
            model = OLIVE(config)
        model.load()
        model.eval()
        old_config = config.copy()

    seg_width, seg_height = image.size
    
    vit_masks = []

    
    cropped_images = []
    for segmentation in segmentations:
        seg = np.array(segmentation)
        if np.sum(seg, axis = None) == 0:
            continue
        else:
        
            mask = np.any(seg != [0, 0, 0], axis=-1)

            if config["crop_image"]:
                img = np.array(image)
                img[~mask] = np.array([255,255,255])
                
                # Find the indices of non-zero elements in the binary mask
                non_zero_indices = np.where(mask)

                # Get the minimum and maximum values along each axis
                min_x, min_y = np.min(non_zero_indices[1]), np.min(non_zero_indices[0])
                max_x, max_y = np.max(non_zero_indices[1]), np.max(non_zero_indices[0])

                img = img[min_y: max_y, min_x: max_x]

                cropped_image = Image.fromarray(np.uint8(img)).convert('RGB')
                cropped_images.append(cropped_image)
            
            vit_masks.append(_get_ViT_mask(mask, seg_height, seg_width, output_height, output_width))
       
    if len(vit_masks) > 0:
        vit_masks = torch.stack(vit_masks, axis = 0)
    imgs = [image] * len(vit_masks) if len(vit_masks) > 0 else [image]
 

    prompts = None
    masks = None
    images = None
    if config['use_retrieval']:
        output, prompts, masks, images = model.generate(vit_masks, imgs, [question], return_retrieved_info=True, cropped_images = cropped_images)
        chat_history.append((question, output))
        retrieval_images = [Image.open(images[0][x]) for x in range(len(images[0]))]
        return chat_history, retrieval_images
        
    else:
        output = model.generate(vit_masks, imgs, [question])
        chat_history.append((question, output))
        return chat_history, None


    

with gr.Blocks(title="Olive", theme=gr.themes.Base()).queue() as demo:
    
    with gr.Row():
        
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    
                    im = gr.ImageEditor(
                        type="pil"
                    )

                    with gr.Row():
                        gallery = gr.Gallery(
                            label="Segmentations", show_label=False, elem_id="gallery"
                        , columns=[3], rows=[1], object_fit="contain", height=200)

                with gr.Column():
                    chatbot = gr.Chatbot(elem_id="chatbot", label="OLIVE Chatbot", height=300)
                    with gr.Row():
                        
                        with gr.Column(scale=8):
                            textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
                        with gr.Column(scale=1, min_width=50):
                            submit_btn = gr.Button(value="Send", variant="primary")
                    retrieval_gallery = gr.Gallery(
                            label="Retrieved Images", show_label=True, elem_id="gallery2"
                        , columns=[5], rows=[1], object_fit="contain", height=100)
   
                    task = gr.Dropdown(["object_classification", "refCOCOg", "ALL"], label="Task",  info="For now object classification/image captioning", value="object_classification")
                    
                    backbone = gr.Dropdown(["llava-hf/llava-1.5-7b-hf", "meta-llama/Llama-2-7b-chat-hf", "gpt2"], label="Decoder Backbone",  info="Backbone Frozen LLM/VLM", value="meta-llama/Llama-2-7b-chat-hf")
                    
                    freeze_llm = gr.Checkbox(label="freeze llm", info="Freeze llm weights", value=True)
                    use_retrieval = gr.Checkbox(label="use retrieval", info="Use retrieval to understand prediction")


        
        im.change(sleep, outputs=[gallery], inputs=im) 

        

    submit_btn.click(fn=generate_predictions, 
                        inputs=[textbox, gallery, task, backbone, use_retrieval, freeze_llm, chatbot],  
                        outputs=[chatbot, retrieval_gallery],  
                        show_progress=True, queue=True)

demo.launch(inbrowser=True)
