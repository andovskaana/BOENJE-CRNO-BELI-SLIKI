# Coloring Black-and-White Images
This project focuses on automatic coloring of black-and-white images using different machine learning models, including pre-trained models, further training of existing architectures, and building custom models from scratch. Each method is tested under different conditions to compare their effectiveness, accuracy, and applicability.

## Contents
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Running Each Part](#running-each-part)
    - [Pre-trained Model with Caffe](#pre-trained-model-with-caffe)
    - [Additional Training with VGG-16](#additional-training-with-vgg-16)
    - [Custom GAN Model with U-Net](#custom-gan-model-with-u-net)
    - [VGG-19 with CUDA and PyTorch](#vgg-19-with-cuda-and-pytorch)
    - [Custom Model with TensorFlow](#custom-model-with-tensorflow)
- [License](#license)

## Project Overview
This project explores different methods for automatic coloring of black-and-white images. Three main approaches are compared:
- Using pre-trained models.
- Further training of existing models using additional datasets.
- Implementing custom models with architectures such as U-Net and GAN.

## Prerequisites
- Python 3.x
- Jupyter Notebook or Google Colab
- Git
- PyTorch, TensorFlow, Keras, OpenCV
- GPU compatible with CUDA (for the PyTorch and CUDA parts)
- Downloading the required CaffeModel files from one of the two links:  
  Official
  ```bash
  https://github.com/richzhang/colorization
  ```
  or
  ```bash
  https://github.com/dhananjayan-r/Colorizer/tree/master/models
  ```

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/andovskaana/BoenjeCrnoBeliSliki.git
   ```
2. Go into the project folder:
   ```bash
   cd BoenjeCrnoBeliSliki
   ```
3. Import and install the required libraries.
4. Create appropriate environments for the platform you are using (Google Colab / PyCharm).

## Running Each Part

### Pre-trained Model with Caffe
This method uses a pre-trained model from the paper “Colorful Image Colorization” by Richard Zhang et al., using the Caffe framework and OpenCV.  
Steps to run:
- Open the `CaffeModel` folder.
- Check that you have the required files: `colorization_deploy_v2.prototxt`, `colorization_release_v2.caffemodel`, and `pts_in_hull.npy`.
- Run the Python script:
  ```bash
  python caffe_colorization.py
  ```

### Additional Training with VGG-16
This part uses the VGG-16 architecture for additional training on datasets such as CIFAR-10, CIFAR-100, and Oxford-IIIT Pet.

Steps to run:
- Open the `GoogleColabs` folder.
- Open the `Treniranje_Svoj_Tenserflow_Model.ipynb` file in Google Colab.
- Provide access to the required datasets.
- Run all steps in `Treniranje_PredTrenirani_Modeli.ipynb` to train the model on the selected dataset.

### Custom GAN Model with U-Net
Implements a custom Generative Adversarial Network (GAN) with a U-Net architecture.

Steps to run:
- Open `Treniranje_PredTrenirani_Modeli.ipynb` in Google Colab.
- Provide access to the required datasets.
- Run all steps in `Treniranje_PredTrenirani_Modeli.ipynb` to train the model on the selected dataset.

### VGG-19 with CUDA and PyTorch
This method is based on a GAN approach with the VGG-19 architecture, using CUDA for accelerated training on NVIDIA graphics cards.

Steps to run:
- Go to the `VGG-ICUDA` folder.
- Make sure you have a CUDA-compatible graphics card and the necessary drivers.
- Open `vgg-icuda-nvidia.py` in PyCharm or your favorite Python IDE.
- Adjust the dataset paths and CUDA settings as needed, then run:
  ```bash
  python vgg-icuda-nvidia.py
  ```

### Custom Model with TensorFlow
A simple Convolutional Neural Network (CNN) using TensorFlow and Keras for basic colorization.

Steps to run:
- Open the `GoogleColabs` folder.
- Open the file `Treniranje_Svoj_Tenserflow_Model.ipynb` in Google Colab.
- Run the steps in `Treniranje_Svoj_Tenserflow_Model.ipynb` to train and test the model using the Oxford-IIIT Pet dataset.

## License
This project is licensed under the MIT License – see the LICENSE file for details.
