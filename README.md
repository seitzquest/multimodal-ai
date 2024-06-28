# Evaluating the Robustness of Scene Graph Generation Models With Known and Unknown Objects Transplantation

This repository contains the code needed for preprocessing the Visual Genome dataset.

## Requirements
There are some requirements needed to run the code in this repository, using conda is the easiest way.
It also suffices to simply use pip
- Create a new conda environment:
```bash
conda create --name multimodal-ai python=3.10
``` 
Note: The python version is 3.10, but any version of python 3 should work.
- Activate the environment:
```bash
conda activate multimodal-ai
```
- Install the required packages:
```bash
pip install -r requirements.txt
```

## Preprocessing

- Run the following commands to preprocess the dataset:
```bash
cd vg_preprocessing
python main.py
```
There are some parameters that can be passed to the script:
- `--modified_directory`: Path to the directory where the modified dataset will be saved.
- `--overlay_image_path`: Path to the overlay image.
- `--seed`: Seed for the random number generator for reproducibility.
- `--num_images`: Number of images to preprocess.