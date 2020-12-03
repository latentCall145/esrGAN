# esrGAN_vBearNinja123

My implementation of the srGAN (https://arxiv.org/pdf/1609.04802v5.pdf) and esrGAN (https://arxiv.org/pdf/1809.00219.pdf) papers, upscaling 32x32 px images into 128x128 px images. I ran the srGAN model using Keras on Google Colab and esrGAN on Kaggle for about 10 hours each on a dataset of turtle images I collected on Google Images (https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/). This means that the models don't work well on upscaling pixel art.

![32x32 LR Image, 128x128 SR Image, 128x128 Ground Truth](/results/32_128.png)

This repo contains a trained esrGAN model (one using dense blocks and one using RRDB blocks described in the esrGAN paper), a trained srGAN model, and a trained esrGAN model that is compatible for Tensorflow 2.2.0, the version that Anaconda currently supports (dense blocks). For fine-tuning your own model, there is an srGAN and esrGAN Jupyter notebook, and you input your low/high-res images in the /train directory.

I also made a script (superResTest.py) to upscale any image 4x if the dimensions of the image is divisible by 32 (e.g 128x128->512x512, 32x64->128x256)
as well as compare the models with each other. I didn't try to hide any artifacts on the edges of the SR images so you'll see faint lines on the upscaled image if you use the script for yourself.

![128x128->512x512](/results/128_512.png)

Here's a picture of the models upscaling the same image.

![srGAN vs. esrGAN vs. esrGAN (RRDB)](/results/comparison.png)
