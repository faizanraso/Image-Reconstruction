from tensorflow import keras
from keras.layers import *
from keras.models import *
from glob import glob 
import cv2
import numpy as np 
from tqdm import tqdm

# define the model
def get_model():
    input = Input(shape= (None, None, 1))
    x = Conv2D(32, 3, activation='relu', padding="same")(input)
    x = Conv2D(64, 3, activation='relu', padding="same")(x)
    x = Conv2D(128, 3, activation='relu', padding="same")(x)
    x = UpSampling2D(2)(x)
    x = Conv2D(64, 3, activation='relu', padding="same")(x)
    x = Conv2D(32, 3, activation='relu', padding="same")(x)
    x = Conv2D(1, 3, activation=None, padding="same")(x)
    x = Activation('tanh')(x)
    x = x*127.5 + 127.5

    model = Model([input], x)
    model.summary()
    # model.compile(optimizer='adam', loss='mse')
    return model
