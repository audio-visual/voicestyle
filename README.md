# voicestyle
 Guess the appearance of an unseen person's face from their unheard voice. A voice-based generation model. Corresponding paper: [TOMM-Voicetyle](https://dl.acm.org/doi/10.1145/3671002)

# Inference

**Install CMPC**  
In order to make it more convenient to use CMPC, we referred to the approach of CLIP: added some methods and provided external APIs.   
Finally, we encapsulated it in the form of a Python package `cmpc2`   
```pip install git+https://github.com/audio-visual/cmpc2.git```

**Download checkpoints and necessary files**  
https://drive.google.com/drive/folders/1U77L9kx95HrpMPBnJGMVb4ShUr3iTR26?usp=drive_link   
stylegan's checkpoint 550000.pt: https://github.com/rosinality/stylegan2-pytorch  
id loss releated checkpoint model_ir_se50.pth: https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view  
put these files to the `weights` folder


**Full pipeline**
```
voice --> (1) find k nearest image prototypes --> (2) get the corresponding image paths
  ↓                                                        ↓
(7) compute CMPC similarity loss               (3) project images to StyleGAN's latent space                                 
  ↑                                                        ↓
(6) generated image <-- (5) feed init to generator <-- (4) aggregate these latent code as init code 
```

**Generate**
```bash
> cd generation
# we uploaded two examples and their releated files, so you can go through the full pipeline
> python pipleline.py
```

> Notice that the first stage is training and testing on images with a resolution of 224x224. However, the StyleGAN(pytorch version) was trained based on 256x256 images. Therefore, in order to obtain better results, we chose to retrain a 256-size version of the CMPC model. (Of course, you can also use the checkpoint of the original 224-size CMPC model and perform upsampling operations in the code)


Important things about GAN inversion(step3):  
Regarding the GAN inversion, we have conducted experiments in advance to test its performance. For the testing process, please refer to: https://github.com/audio-visual/investigate_stylegan.   
The test results show that the low-quality, excessively large head angle, and misaligned images in the voxceleb dataset (although they can be aligned through post-processing, but due to low resolution, the post-processing results are very bad) are not suitable for gan inversion.    

Therefore, we adopted a compromise solution, that is, replacing the faces in voxceleb videos with the same faces in vggface(filterd by https://github.com/cmu-mlsp/reconstructing_faces_from_voices, named VGG_ALL_FRONTAL). However, this also has limitation, that is, vggface does not have video-level tags, and the appearance, makeup, and expressions of the same person will vary greatly (as shown in the below). This may reduce model performance. 

Examples from VGG_ALL_FRONTAL 'Luke_Mitchell'
![image](https://github.com/audio-visual/voicestyle/assets/110716367/4361934c-c869-4c77-ad3c-0f3e0a53cc4f)

**input voice example1:** 

https://github.com/audio-visual/voicestyle/assets/110716367/a3b602f7-84a0-4f7a-947e-f7bfcbe77eea 

You can try to imagine it first :)

 generated face  |  real face
:-------------------------:|:-------------------------: 

 <img src='https://github.com/audio-visual/voicestyle/blob/main/results/id10724_DP5gCFoWgGI.png' width='256px'/>  | <img src='https://github.com/audio-visual/voicestyle/assets/110716367/7cc5d439-8360-4a5e-8f94-dcd19588867e' width='256px'/> 

-------------------------------------------------------------------------------------------------------------------------------------------------

**input voice example2:**  

https://github.com/audio-visual/voicestyle/assets/110716367/305f6191-ba66-4e8f-9f2f-437384c5fadd

 generated face  |  real face
:-------------------------:|:-------------------------: 

 <img src='https://github.com/audio-visual/voicestyle/blob/main/results/id10845_0RsOMTn-DSM.png' width='256px'/>  | <img src='https://github.com/audio-visual/voicestyle/assets/110716367/7a2eeddf-8193-4ebc-ae32-7833382277b6' width='256px'/> 

## Citation

```
@article{10.1145/3671002,
author = {Chen, Wuyang and Zhu, Boqing and Xu, Kele and Dou, Yong and Feng, Dawei},
title = {VoiceStyle: Voice-Based Face Generation via Cross-Modal Prototype Contrastive Learning},
year = {2024},
issue_date = {September 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {20},
number = {9},
issn = {1551-6857},
url = {https://doi.org/10.1145/3671002},
doi = {10.1145/3671002},
journal = {ACM Trans. Multimedia Comput. Commun. Appl.},
month = sep,
articleno = {279},
numpages = {23},
keywords = {Cross-modal representation learning, contrastive learning, StyleGAN, face generation}
}
```
