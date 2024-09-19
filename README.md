# Hematology Segmentation
## Overview:
This repository contains implementations of deep learning models for segmenting hematology images, focusing on blood cell analysis. It uses both CNN and transformer-based approaches, integrating the nnUNet framework for model adaptability and high-performance segmentation. The main goal is to automate cell identification and segmentation in hematological images, improving efficiency in clinical diagnostics.

## Installation:
Clone the repository:\
`git clone https://github.com/chloe-nguyenminh/Hematology-Segmentation.git`

Create a conda virtual environment:\
`conda create --name hema-seg python=3.10`\
`conda init`\
`conda activate hema-seg`

Install Pytorch:\
`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

Install the Repository:\
`cd TriALS`\
`pip install -e .`

## Steps to reproduce the result
## I. Data Preparation:
, The below steps must be run to adapt the Hematology dataset to the nnU-Net V2 requirement for data preparation:

1. Download and save the dataset into your `data/Original_data` directory
2. Export global environment variables\
`export dataset_name='Dataset019_Hema`\
`export source="data/Original_data`\
`export nnUNet_raw=<path-to>/data/nnUNet_raw_data_base/`\
`export nnUNet_preprocessed=<path-to>/data/nnUNet_preprocessed/`\
`export nnUNet_results=<path-to>/data/nnUNet_results/`

3. Convert the Hematology dataset into nnU-Net format: \
For each `folder` of training and testing images:\
`python preprocess_imgs.py --img_dir=<path_to>/folder_name/`\
`python Dataset019_Hema.py --img_dir=<path_to>/preprocessed_folder/`

New `preprocessed_folder_name` will be created corresponding to each folder to avoid overwriting the original data for more robust development. Two .json files will be created, one containing the original size of the images and another containing the metadata of the dataset. This information can be changed in the `Dataset019_Hema.py` if needed.

5. Experiment planning and preprocessing:\
`nnUNetv2_plan_and_preprocess -d 019 --verify_dataset_integrity`


## II. Model Training
To train the UNet model:\
`nnUNetv2_train 019 2d -tr nnUNetTrainer -p nnUNetPlans --npz `

To train the SAMed model:\
`CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 019 2d_p512 -tr nnUNetTrainerV2_SAMed_h_r_4_100epochs --npz`

If there are errors with Torchynamo, or died background worker, the following setup can be added before each command: \
`TORCH_LOGS="+dynamo" TORCHDYNAMO_VERBOSE=1 nnUNet_n_proc_DA=0`

To determine the best configuration after running full cross-validation:\
`nnUNetv2_find_best_configuration Dataset019_Hema -c 2d`

## III. Inference
To run inference which incorporate both nnU-Net inference and postprocessing accommodated for the Hematology dataset:\
`python inference.py --input_path=<path_to_input_images> --output_path=<path_to_output_images>`

To visualize the segmentation masks overlaid onto the corresponding original images and a side-by-side comparison:\
`python visualize.py --input_path=<path_to_input_images> --output_path=<path_to_output_images>`

## IV. Acknowledgement:
The implementation of nnU-Net is adapted from https://github.com/MIC-DKFZ/nnUNet. The implementation of the embedded SAM training functionalities within the nnU-Net framework is adapted from https://github.com/xmed-lab/TriALS.
