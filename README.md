# Image Reconstruction Using CNN Based Upsampling

This project involves a Python program that performs downsampling of images, converts them to the YUV color space, and then upsamples them using a trained Convolutional Neural Network (CNN) model.

## Prerequisites

To install all required packages run the following command: <br/>
```pip install -r requirements.txt``` 

## Usage

1. Define the input image at the top of the main.py file
2. Run ```python3 main.py```

## Demo

<p align="center">
  <img src="https://user-images.githubusercontent.com/59986120/230701888-9a6a535d-ade5-42ec-89c6-33525082dfac.jpg" height="300"/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://user-images.githubusercontent.com/59986120/230702056-93043735-ccfe-4fb0-a274-e26ffb9f9092.png" height="300"/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://user-images.githubusercontent.com/59986120/230702104-c70b3328-70a5-4d40-af01-41fe4d648f90.png" height="300"/>
</p>

The original (left) is converted to the YUV and donwsampled (middle). The Y channel is downsampled by a factor of 2 as it contains key image data while the U and V channels are downsampled by a factor of 4. The individual channels are then upsampled, using their corresponding CNN trained models. The upsampled channels are then concatenated to form the reconstructed image (right)
