# voicestyle
 Guess the appearance of an unseen person's face from their unheard voice. A Sound Based Generation Model

# Inference

In order to make it more convenient to use CMPC, we referred to the approach of CLIP: added some methods and provided external APIs.   
Finally, we encapsulated it in the form of a Python package `cmpc2` 

**Install CMPC**
```pip install git+https://github.com/audio-visual/cmpc2.git```

**Download checkpoints**
https://github.com/rosinality/stylegan2-pytorch

full pipeline:
```
voice --> (1) find k nearest image prototypes --> (2) get the corresponding image paths
  ↓                                                        ↓
(7) compute CMPC similarity loss               (3) project images to StyleGAN's latent space                                 
  ↑                                                        ↓
(6) generated image <-- (5) feed init to generator <-- (4) aggregate these latent code as init code 
```
> Notice that the first stage is training and testing on images with a resolution of 224x224. However, the StyleGAN(pytorch version) was trained based on 256x256 images. Therefore, in order to obtain better results, we chose to retrain a 256-size version of the CMPC model. (Of course, you can also use the checkpoint of the original 224-size CMPC model and perform upsampling operations in the code)


Important things about GAN inversion(step3):  
Regarding the GAN inversion, we have conducted experiments in advance to test its performance. For the testing process, please refer to: https://github.com/audio-visual/investigate_stylegan. The test results show that the low-quality, excessively large head angle, and misaligned images in the voxceleb dataset (although they can be aligned through post-processing, but due to low resolution, the post-processing results are very bad) are not suitable for gan inversion.  

Therefore, we adopted a compromise solution, that is, replacing the faces in voxceleb videos with the same faces in vggface(filterd by https://github.com/cmu-mlsp/reconstructing_faces_from_voices). However, this also has drawback, that is, vggface does not have video-level tags, and the appearance, makeup, and expressions of the same person will vary greatly (as shown in the below). This may reduce model performance. 

TODO 找示例图

id loss model_ir_se50.pth: https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view

