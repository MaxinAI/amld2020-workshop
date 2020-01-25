# Generative Modeling for Computer Vision - Applied Machine Learning Days 2020
This repository contains [workshop](https://appliedmldays.org/workshops/generative-modeling-for-computer-vision?fbclid=IwAR1cT39lJwipvxbaLZQz7ynFcSq112k00aKxNn_aedZFmbXVWcjaxpwgQXg) material for Applied Machine Learning Days 2020 conference.

# Authors:
Anzor Gozalishvili - https://www.linkedin.com/in/anzor-gozalishvili

Sandro Barnabishvili - https://www.linkedin.com/in/sandrobarna

# Workshop Summary:
The goal of generative machine learning is to learn data distribution in order to generate new data points during the inference. Classical examples of generative algorithms are generating text from the voice and vice versa, applying artistic styles of famous painters to pictures, creating fake faces of people and many more. This workshop will be a hands-on tutorial on deep auto-encoders and variational auto-encoders – one of the most popular unsupervised generative algorithms. We will go through a lot of Illustrative code examples to point out advantages and disadvantages of using either of these models.

# Workshop Outcome
Participants will get strong intuition about various types of auto-encoders and hands-on experience to build and adjust one to their needs.

# Difficulty:
Intermediate level

## General instructions:
- By cloning the repo, you will have the notebooks needed during the workshop. All these notebooks can be run using [Google Colab](https://colab.research.google.com/) (Ready to go)
- To have a working environment for the code install the packages from the `requirements.txt`. (If conda or pip doesn't install packages directly you have to install them one by one).

## Running the notebooks:
During the workshop we do a small presentation first to give some intuitions about autoencoders and variational autoencoders. Then we run notebooks:
- [ae_vae_mnist.ipynb](https://github.com/MaxinAI/amld2020-workshop/blob/master/ae_vae_mnist.ipynb) - we compare vanilla autoencoder and variational autoencoder architectures and their latent spaces. We show that variational autoencoder forms better (continuous with smooth transitions) latent space. We show entire 2D latent space on MNIST example and some other examples to demonstrate latent space features.
- [cnn_vae_celeba.ipynb](https://github.com/MaxinAI/amld2020-workshop/blob/master/cnn_vae_celeba.ipynb) - we use variational autoencoder with convolutional encoder&decoder and train on customized [CelebA](https://www.kaggle.com/jessicali9530/celeba-dataset) dataset from Kaggle (cropped and aligned on faces). We do some face feature arithmetics such as adding sunglasses and smile to any face. We show an example of perceptual loss added on standard ELBO to get smooth reconstructions on first iterations.
- [vq_vae_mnist (experimental).ipynb](https://github.com/MaxinAI/amld2020-workshop/blob/master/vq_vae_mnist%20(experimental).ipynb) - in this experimental version of notebook we showcase an example of Vector Quantized Variational Auto Encoders on MNIST dataset. We try to use only 10 embeddings in quantizer Codebook to get disentagled latent representations for each MNIST digit. (It doesn't work for some digits and it's seems to be an interesting open problem to think about)
- [vq_vae_celeba (experimental).ipynb](https://github.com/MaxinAI/amld2020-workshop/blob/master/vq_vae_celeba%20(experimental).ipynb) - in this experimental version of notebook we showcase an example of Vector Quantized Variational Auto Encoders on CelebA (cropped and face aligned) dataset. We try to do the same simple arithmetic of face attributes which doesn't work per image but works for averaged faces (averaging is done in latent space). There are also several open questions and probelms to try and experiment here on your own.

# References:
- Original VAE paper https://arxiv.org/abs/1312.6114
- CelebA dataset http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html#sparse-autoencoder
- Weight initialization https://arxiv.org/abs/1502.01852
- More upsampling techniques https://arxiv.org/abs/1609.05158
- Vector-Quantized Auto-Encoders https://arxiv.org/abs/1711.00937
- Perceptual Loss https://arxiv.org/abs/1603.08155

# Interesting Papers:
- https://arxiv.org/abs/1906.00446
- https://github.com/rosinality/vq-vae-2-pytorch
- https://www.ncbi.nlm.nih.gov/pubmed/29218871
- https://arxiv.org/abs/1910.10942
- http://dafx2019.bcu.ac.uk/papers/DAFx2019_paper_20.pdf
- https://github.com/yjlolo/vae-audio
- https://arxiv.org/abs/1903.07137
- https://www.aclweb.org/anthology/P19-1199/
- https://arxiv.org/abs/1810.01112
- https://arxiv.org/abs/1910.03957
- https://arxiv.org/abs/1811.10276
- https://arxiv.org/pdf/1806.03182.pdf
