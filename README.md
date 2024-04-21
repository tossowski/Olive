# Object Level In-Context Visual Embeddings (OLIVE)

This repository contains the code for the paper "Object Level In-Context Visual Embeddings (OLIVE)". The repo has code to train the models described in the paper, which can be achieved by following the instructions below.

![Architecture of OLIVE](images/architecture.svg)

* The model allows for free form selection of areas of the image for reasoning.

* Our model does not prepend the standard array of image patch features to the LLM decoder input to speed up training and inference and be memory-friendly. This loses some information from the background, but we have found that the model can still understand object level detail and display scene content awareness.

* We experiment with object level retrieval to generalize to unseen visual concepts.

More details can be found in the paper. We also illustrate example usage in the gif below:



### Setup
We highly recommend setting up an anaconda environment with the required dependencies using the code below:

```python
conda create --name OLIVE
conda activate OLIVE
pip install -r requirements.txt
```

### Config File
This is the most important file which controls the experiments. We have several examples in the `configs` folder for different experiments.


### Datasets
Our experiments in our paper mainly involve COCO object detection and refCOCOg datasets. You only need to download COCO images for this. However, we also experiment with domain adaptation on medical images using the Chest X-Ray (CXR8) Dataset. You can download all of this COCO and medical data using our setup script. Make sure to set the `DATA_FOLDER` path in the config file first. You will have to download the **BBox_List_2017.csv**, **train_val_list.txt**, and **test_list.txt** files from their website: https://nihcc.app.box.com/v/ChestXray-NIHCC

```
python setup/setup.py
```

Some unpublished experiments also use the visual genome and GRIT datasets. The final directory structure should look like this:

```
data
|----COCO2017
    |----train
        |----000000009.jpg
        |----000000014.jpg
                ⋮
    |----val
        |----000000009.jpg
        |----000000014.jpg
                ⋮
    |----refCOCOg
        |----refs(google).p
|----CXR8
    |----BBox_List_2017.csv
    |----test_list.txt
    |----train_val_list.txt
    |----images
        |----00000887_003.png
        |----00000888_000.png
                ⋮
|----vg
    |----VG_100K
    |----VG_100K_2
|----GRIT
    |----coyo_snappy0.parquet
    |----coyo_snappy1.parquet
                ⋮
    |----images
        |----0000

```
### Retrieval Set Preparation
We have a separate script to prepare the retrieval set to train models with retrieval capability. To prepare the retrieval set, set the task to be `object_classification` in the config.yaml file and run the following script:
```
python retrieve.py --train
```
This will create a .pkl file in a folder called `retrieval`.

To test the performance of retrieval only methods, you can then run
```
python retrieve.py --test
```
### Training
After setting up the config file in `configs/config.yaml`, you can train a model using that config. For example, to do object classification, you may try

```python
python main.py --train --config configs/object_classification.yaml
```

Intermediate checkpoints are stored in the checkpoints folder, where you can also see the loss over parameter updates graph.

### Testing and Demo
We have some evaluation scripts for the object classification and region description tasks. For most testing, you do not need to change the `config.yaml` file from the training configuration. However, for the domain adaptation experiment, make sure to set the task to the downstream task (e.g. `medical_object_classification`), and type in the path of the model checkpoint you want to load for `load_model_path`.

You can test the model (after setting up the `config.yaml`) using:

```python
python main.py --test
```

After training, you can also test the model qualitatively using our demo notebook. Run the cells in the `demo.ipynb` notebook to open the gradio interface:

![Gradio Interface](images/example.png)

To use the demo, make sure to put `[obj]` in your chat message to refer to your selected object. This will be replaced by the object feature during input preparation.