# ML Seismic Interpretation 🌍
### Repository containing code on ML/DL Assisted Seismic Interpretation without human annotation

--------------------

## Structure
```bash
< PROJECT ROOT >
   |
   |-- dl_methods/          
   |    |-- FaultSeg3D/
   |    |    |-- FaultSeg3D_Test.ipynb
   |    |    |-- faultseg3d_test.py                   # Wrapper functions for testing out FaultSeg3D
   |    |    
   |    |-- dl_workflow.ipynb                         # Model Training and Prediction
   |    |
   |    |-- networks/
   |    |    |-- vit_seg_configs.py                   # ViT Transformer UNet transformer configuration
   |    |    |-- vit_seg_modeling_resnet_skip.py      # ViT Transformer UNet ResNet Block
   |    |    |-- vit_seg_modelling.py                 # ViT Transformer UNet model
   |    |    |-- vit_seg_gan_two_stage.py             # ViT Transformer two-stage model + Patch Discriminator
   |    |
   |    |-- losses.py                                 # Loss Functions
   |    |-- dataloader.py                             # Seismic Dataloader
   |    |-- augmentation.py                           # Data Augmentation Functions
   |
   |-- requirements.txt                               # Project Dependencies
   |-- environment.yml                                # ENV Configuration (conda)
   |
   |-- labelled_npy_data/
   |    |-- synthetic_3d.py                           # Module for generating 3D synthetic seismic reflection data
   |
   |-- README.md                                      # This file
   |-- .gitignore                                     # gitignore
   |-- images/                                        # Images (.png, .jpg)
   |
   |-- ************************************************************************
```

## Data
### Publicly Available Field Data
You can download the preprocessed .npy files (reshaped into (timeslice, crossline, inline)) containing the seismic and facies-labelled arrays [here](https://drive.google.com/drive/folders/1hXBmz4orsGYDi6cQ-eWF69NPFRklv8HV?usp=sharing), or import the data from their original source:

| Survey Name | Source |
| ----------- | ------ |
| Netherlands F3 Block | [https://github.com/yalaudah/facies_classification_benchmark](https://github.com/yalaudah/facies_classification_benchmark) |
| Parihaka | [https://www.aicrowd.com/challenges/seismic-facies-identification-challenge](https://www.aicrowd.com/challenges/seismic-facies-identification-challenge)|
| Penobscot | [https://zenodo.org/record/3924682](https://zenodo.org/record/3924682) |

Unlabelled .segy files containing only the seismic reflection data can be downloaded from [here](https://drive.google.com/drive/folders/1vF_7ACryT9wRGGGAOLfcJtGNyFvKO82J?usp=sharing)

### Synthetic Data

<code>synthetic_3d.py</code> contains functions for generating 3d labelled synthetic seismic data with faults, folds, salt and channel features, and facies variation. Channel features are generated with the help of the [meanderpy](https://github.com/zsylvester/meanderpy) package.

<p align="center">
	<img src="https://github.com/gems-hcl4517/ML-Seismic-Interp/blob/main/images/example_generated_3d_seismic.png?raw=true" width="600"/>
</p>


## Deep Learning Workflow (Semi-Supervised Learning & Domain Adaptation)
A Vision Transformer UNet model is implemented based on the structure provided by the [TransUNet](https://github.com/Beckschen/TransUNet) repository, and is trained using generated 3D synthetic data from <code>synthetic_3d.py</code>. The model is further expanded to contain a two-stage structure, with the first stage only outputting segmentation for faults and channels, to counter for class imbalance. 

<p align="center">
	<img src="https://github.com/gems-hcl4517/ML-Seismic-Interp/blob/main/images/dl_model_structure_full.png?raw=true" width="600"/><br>
	<sub><sup>Deep Transfer Learning structure. a) Schematic of the Transformer layer. b) Transformer UNet structure. Modified from Chen et al 2021. c) Two-stage framework. d) Semi-supervised Learning flowchart. e) Adversarial Training flowchart.</sup></sub>
</p>

Semi-supervised training has been employed via a domain-specific augmentation of faulting and folding, and other augmentations such as horizontal flipping, and enlargement, on unlabelled field data. A Patch Discriminator is used to improve the performance further. Number of input channels is set to be configurable for 2.5D learning (i.e. prediction of an xline/inline slice from a multi-channel input also containing neighbouring slices).

<p align="center">
	<img src="https://github.com/gems-hcl4517/ML-Seismic-Interp/blob/main/images/transfer_learning_predictions.png?raw=true" width="600"/><br>
	<sub><sup>Prediction outcomes of a seen unlabelled area with a high density of faults, from the Thebe seismic subset. a) Original seismic timeslice at x = 120. e) Original seismic inline at x = 100. i) Original seismic crossline at x = 120. b), f), j) Fusion of argmax predictions from the 2.5D Two-stage TransUNet GAN model in both inline and crossline directions for the presented slices respectively. c), g), l) Respective seismic slices overlaid on top by fault predictions from FaultSeg3D. d), h), k) Respective seismic slices overlaid on top by fusion of probability prediction of faults from the 2.5D Two-stage TransUNet GAN model in both inline and crossline directions.</sup></sub>
</p>

<code>dl_workflow/</code> also contains files and notebook showcasing the application of [FaultSeg3D](https://github.com/xinwucwp/faultSeg) on the seismic datasets as comparison. Download the pretrained FaultSeg3D model files from [here](https://drive.google.com/drive/folders/1L_Trsyo2ll8i7uk16t3Sp_o1xK3iePLa?usp=sharing).

---------------

