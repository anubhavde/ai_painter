#### **octo-monet-garbanzo**

# **Generative Adversarial Network that generates Monet-style images**

We recognize the works of artists through their unique style, such as color choices or brush strokes. The *“je ne sais quoi”* of artists like Claude Monet can now be imitated with algorithms thanks to generative adversarial networks (GANs).

Computer vision has advanced tremendously in recent years and GANs are now capable of mimicking objects in a very convincing way. But creating museum-worthy masterpieces is thought of to be, well, more art than science. So we will use (data) science, in the form of GANs to trick classifiers into believing we’ve created a true Monet.

A GAN consists of at least two neural networks: a generator model and a discriminator model. The generator is a neural network that creates the images. For our competition, you should generate images in the style of Monet. This generator is trained using a discriminator.

The two models work against each other, with the generator trying to trick the discriminator, and the discriminator trying to accurately classify the real vs. generated images.

### **The Dataset**

The dataset contains four directories: monet_tfrec, photo_tfrec, monet_jpg, and photo_jpg. The monet_tfrec and monet_jpg directories contain the same painting images, and the photo_tfrec and photo_jpg directories contain the same photos.

The monet directories contain Monet paintings. These images will be used to train our model.

The photo directories contain photos. AI generated Monet-style images will be added here. Other photos outside of this dataset can be transformed.

Note: Monet-style art can be created from scratch using other GAN architectures like DCGAN.

- monet_jpg - 300 Monet paintings sized 256x256 in JPEG format
- monet_tfrec - 300 Monet paintings sized 256x256 in TFRecord format
- photo_jpg - 7028 photos sized 256x256 in JPEG format
- photo_tfrec - 7028 photos sized 256x256 in TFRecord format

### **SetUp**

The notebook utilizes a CycleGAN architecture to add Monet-style to photos. We will be using the TFRecord dataset. Import the necessary packages and change the accelerator to TPU if you are running this notebook from Kaggle.

