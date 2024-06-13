Based on the working principles of the deep learning Diffusion model and in combination with the project requirements, the following changes are made to the traditional model processing flow:
Firstly, the sample resolution we use is 512x512. Training directly with such a dataset requires high computing power, so it is more reasonable to adopt a cascade structure. We first generate a low-resolution image of 128x128 from random noise, and then super-resolve the adjustment of the low-resolution image to generate a high-resolution image of 512x512. Secondly, in each iteration of the diffusion model generation process, we introduce low-dose CT images and solve the MAP estimation problem to ensure that the generated images have good likelihood with the input low-dose CT images. Finally, based on the Gamma evaluation method, we evaluate the accuracy of the denoised images and FullDose images.
train:

val:
 python dcm_128_denoising_dcm.py -c ./config/Dn_liver_128.yaml        
 python dcm_128_512_denoising_dcm.py -c ./config/Dn_liver_128_512.yaml
gamma：
 python -c gamma.py
