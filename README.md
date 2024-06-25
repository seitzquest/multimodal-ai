# Evaluating the Robustness of Scene Graph Generation Models With Known and Unknown Objects Transplantation

This repository contains the code needed for preprocessing the Visual Genome dataset.

## Requirements
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
- TODO: Add instructions on how to obtain VG Dataset
- Run the following command to preprocess the dataset:
```bash
python main.py
```
There are some parameters that can be passed to the script:
- `--source_directory`: Path to the directory containing the Visual Genome dataset.
- `--modified_directory`: Path to the directory where the modified dataset will be saved.
- `--overlay_image_path`: Path to the overlay image.
- `--seed`: Seed for the random number generator for reproducibility.