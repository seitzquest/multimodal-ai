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
- `--patch_strategy`: Strategy for finding patch. Can be 'minimal', 'maximal' or 'random'
- `--correlate_overlay`: Overrides the overlay image and patch strategy by using a correlated object
- `--visualize-bb`: Visualize bounding boxes


## Evaluation
In `baseline_models/` you can find our adapted code for the two evaluated SGG models RelTR and SpeaQ.
Please consult the local `README.md` file for their setup and how to download the VG dataset. Make sure that you also install the packages from `requirements.txt` in their respective conda environments.

Additionally we use SAM for segmentation, which can be downloaded [here](https://huggingface.co/ybelkada/segment-anything/blob/main/checkpoints/sam_vit_h_4b8939.pth)

### Setup Tips for RelTR
For RelTR, we encountered a conda error with opencv when trying to add it to the python 3.6 conda environment. As described in [here](https://stackoverflow.com/a/63752514), fixing the version using `pip install opencv-python==3.4.13.47` resolved the issue.

Later on, we decided to use python 3.10 instead of the 3.6 in order to run the code on a modern GPU as it is required for installing pytorch nightly. The only adaptation that was necessary, was installing CoCo using `pip install pycocotools` instead of the CoCo Dataset PythonAPI `pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'`. Using 3.10 also resolved the version issue with opencv described above.
