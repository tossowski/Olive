import yaml
import os
import sys
sys.path.insert(0, os.path.abspath(os.getcwd()))

import subprocess
import numpy as np
import os
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.COCO_Captions import COCOCaptionsDataset
import json
from pycocotools import mask
import skimage.measure
import math
import os
import yaml


with open("configs/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

DATA_FOLDER = config['DATA_FOLDER']

train_image_path = os.path.join(DATA_FOLDER, "COCO2017")

os.makedirs(train_image_path, exist_ok=True)
os.chdir(train_image_path)

# Download Data

# Object Classification Data
subprocess.run(["wget", "http://images.cocodataset.org/zips/train2017.zip"])
subprocess.run(["wget", "http://images.cocodataset.org/zips/val2017.zip"])
subprocess.run(["wget", "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"])
subprocess.run(["unzip", "train2017.zip"])
subprocess.run(["unzip", "val2017.zip"])
subprocess.run(["unzip", "annotations_trainval2017.zip"])
subprocess.run(["rm", "train2017.zip"])
subprocess.run(["rm", "val2017.zip"])
subprocess.run(["rm", "annotations_trainval2017.zip"])


# RefCOCOg (Referring Expression Generation)
subprocess.run(["wget", "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip"])
subprocess.run(["unzip", "refcocog.zip"])
subprocess.run(["rm", "refcocog.zip"])


# Medical Chest XRays
import urllib.request

# URLs for the zip files
links = [
    'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
    'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
    'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
	'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
    'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
	'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
	'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
    'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
	'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
	'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
	'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
	'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
]

import tarfile
os.makedirs(os.path.join(DATA_FOLDER, "CXR8"), exist_ok=True)
for idx, link in enumerate(links):
    fn = 'images_%02d.tar.gz' % (idx+1)
    location = os.path.join(DATA_FOLDER, "CXR8", fn)
    print('Downloading '+fn+'...')
    urllib.request.urlretrieve(link, location)  # download the zip file

    with tarfile.open(location, "r:gz") as tar:
        # Extract all contents to the specified directory
        tar.extractall(path=os.path.join(DATA_FOLDER, "CXR8"))

    subprocess.run(["rm", location])


for split in ["train", "val"]:
    for patch_size in [16, 24]:
        output_width = patch_size
        output_height = patch_size
        print(f"Creating {split} segmentation instruction data with {patch_size}x{patch_size} patches ...")
        include_segmentations = True
        task = "object_detection"
        dataset_val = COCOCaptionsDataset(os.path.join(DATA_FOLDER, "COCO2017", f"{split}2017"), 
                                        os.path.join(DATA_FOLDER, "COCO2017", "annotations", f"captions_{split}2017.json"), 
                                        os.path.join(DATA_FOLDER, "COCO2017", "annotations", f"instances_{split}2017.json"),
                                        f"{split}")

        #dataset_val = COCOCaptionsDataset(f"/data/ossowski/COCO2017/{split}/data", f"/data/ossowski/COCO2017/annotations/captions_{split}2017.json", f"/data/ossowski/COCO2017/{split}/labels.json", f"{split}")
        train_loader = DataLoader(dataset_val, 1, shuffle=False, num_workers=2)

        segment_string = "with_segmentations" if include_segmentations else "no_segmentations"
        save_path = os.path.join(DATA_FOLDER, "COCO2017", "instruction_data", f"supervised_{segment_string}_{split}_{task}_{output_width}x{output_height}.json")

        #save_path = f"/data/ossowski/COCO2017/instruction_data/supervised_{segment_string}_{split}_{task}_{output_width}x{output_height}.json"
        print(f"Will save data to {save_path}")
        data = []

        for k, batch in enumerate(tqdm(train_loader)):
            d = {}
            image_id = batch['image_id'][0].item()
            height = dataset_val.image_id_dict[image_id]['height']
            width = dataset_val.image_id_dict[image_id]['width']
            segmentations = dataset_val.image_id_dict[image_id]['segmentations']
            segmentation_labels = dataset_val.image_id_dict[image_id]['segmentation_labels']
            bboxes = dataset_val.image_id_dict[image_id]['bboxes']
            assert(len(segmentations) == len(segmentation_labels))
            masks = []
            original_segmentations = []
            final_labels = []
            class_types = []
            for i, segmentation in enumerate(segmentations):
                if type(segmentation) == list:
                    lbl = segmentation_labels[i]
                    
                    rles = mask.frPyObjects(segmentation, height, width)
                    rle = mask.merge(rles)
                    m =  mask.decode(rle).astype(bool)

                    rle['counts'] = rle['counts'].decode('ascii')
                    original_segmentations.append(rle)
                    
                    pooled_mask = skimage.measure.block_reduce(m, block_size=(math.floor(height / output_height), math.floor(width / output_width)), func=np.max)
                

                    result_height, result_width = pooled_mask.shape

                    # If the result is smaller than 16x16, pad it with zeros
                    if result_height < output_height or result_width < output_width:
                        pad_height = output_height - result_height
                        pad_width = output_width - result_width
                        pooled_mask = np.pad(pooled_mask, ((0, pad_height), (0, pad_width)), mode='constant')
                    
                    if result_height > output_height or result_width > output_width:
                        pooled_mask = pooled_mask[:output_height, :output_width]

                    assert pooled_mask.shape == (output_height,output_width)
                    rle_mask = mask.encode(np.asfortranarray(pooled_mask))
                    rle_mask['counts'] = rle_mask['counts'].decode('ascii')
                    masks.append(rle_mask)
                    final_labels.append(segmentation_labels[i])

            path_to_image = os.path.join(dataset_val.image_path, f"{str(image_id).zfill(12)}.jpg")
            info = dataset_val.image_id_dict[batch['image_id'][0].item()]
            caption = info['captions'][0]
            d['id'] = str(image_id).zfill(12)
            d['image'] = path_to_image
            d['bboxes'] = bboxes
            d['class_types'] = class_types
            if include_segmentations:
                d['segmentations'] = masks
                d['segmentation_labels'] = final_labels
                d['original_segmentations'] = original_segmentations
                assert(len(masks) == len(final_labels))
                assert(len(original_segmentations) == len(final_labels))

            if task == "object_detection":
                n_objects = len(list(set(segmentation_labels)))
                if n_objects == 1:
                    d['conversations'] = [{"from": "human", "value": "<image>\nWhat objects are in the image?"}, {"from": "gpt", "value": f"The image contains a {', '.join(list(set(segmentation_labels)))}"}]
                else:
                    d['conversations'] = [{"from": "human", "value": "<image>\nWhat objects are in the image?"}, {"from": "gpt", "value": f"The image contains {', '.join(list(set(segmentation_labels)))}"}]

            elif task == "image_captioning":
                
                d['conversations'] = [{"from": "human", "value": "<image>\nCan you write a one sentence caption for this image?"}, {"from": "gpt", "value": f"{caption}"}]
            data.append(d)

        os.makedirs(os.path.join(DATA_FOLDER, "COCO2017", "instruction_data"), exist_ok=True)
        json.dump(data, open(save_path, "w"))
