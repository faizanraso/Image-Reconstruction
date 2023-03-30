from tensorflow import keras
from keras.layers import *
from keras.models import *
from glob import glob 
import cv2
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Convert RGB to YUV
def rbg_to_yuv(r, g, b):
    y = np.zeros(r.shape, dtype=np.float32)
    u = np.zeros(r.shape, dtype=np.float32)
    v = np.zeros(r.shape, dtype=np.float32)

    y = 0.299*r + 0.587*g + 0.114*b
    u = -0.147*r - 0.289*g + 0.436*b + 128
    v = 0.615*r - 0.515*g - 0.100*b + 128

    return y, u, v


# Floor values to get the integer part values will be between 0 and 255
def floor_values(y, u, v):
    Y = np.asarray(np.floor(y), dtype='int')
    U = np.asarray(np.floor(u), dtype='int')
    V = np.asarray(np.floor(v), dtype='int')
    return Y, U, V


# Downsample the image by 2 for Y and 4 for U and V
def downsample(y, u, v):
    Y = y[0::2, 0::2]
    U = u[0::4, 0::4]
    V = v[0::4, 0::4]
    return Y, U, V

def resize_image(image, new_height, new_width):
    height, width = image.shape[:2]
    height_ratio = new_height / float(height)
    width_ratio = new_width / float(width)

    # Resize the image using the scaling factors
    resized_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    for i in range(new_height):
        for j in range(new_width):
            y = int(i / height_ratio)
            x = int(j / width_ratio)
            resized_image[i,j,:] = image[y,x,:]
    
    return resized_image

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
    return model

def get_data():
    x = []
    y = []
    for img_dir in tqdm(glob('images/train/000*.png')):
        img = cv2.imread (img_dir)
        img = resize_image(img, 648, 1116)
        B, G, R = cv2.split(img)
        Y, U, V = rbg_to_yuv(R, G, B)
        Y, U, V = floor_values(Y, U, V)
        
        y_channel = Y[:,:]
        y_out = y_channel

        Y, U, V = downsample(Y, U, V)
        y_in = Y

        print(y_in.shape, y_out.shape)
        x.append(y_in)
        y.append (y_out)
    
    x = np.array(x)
    y = np.array(y)

    return x, y

def train():
    model = get_model()
    x, y = get_data()
    print (x.shape, y.shape)

    # plt.subplot (211)
    # plt.imshow(x[0], cmap='gray')
    # plt.subplot(212)
    # plt.imshow(y[0], cmap='gray')
    # plt.show()

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    loss = 'mse'
    model.compile(optimizer=optimizer, loss=loss)

    save_model_callback = keras.callbacks.ModelCheckpoint(
        filepath='model/model.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        save_freq='epoch',
    )

    tb_callback = keras.callbacks.TensorBoard(
        log_dir='./Graph',
        histogram_freq=0,
        write_graph=True,
        write_images=True,
    )

    batch_size = 4
    epochs = 100
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), validation_split=0.1, callbacks=[save_model_callback, tb_callback])


def load_model():
    model = load_model('model/model.h5')
    img = cv2.imread ('images/valid/0801.png')
    img_ycrcb = cv2. cvtColor (img, cv2.COLOR_BGR2YCrCb)
    y_channel = img_ycrcb[:, :, 0]
    y_in = cv2.resize(y_channel, (256, 256), interpolatio=cv2.INTER_AREA)
    y = np.expand_dims (y_in, axis=0)
    
    # apply preprocessing here
    y = y/127.5 - 1
    y = np.expand_dims (y, axis=3)

    y_upsampled = model.predict (y)
    print (y_upsampled.shape)
    plt.subplot(211)
    plt.imshow(y[0], cmap='gray') 
    plt.subplot (212) 
    plt.imshow(y_upsampled[0], cmap='gray') 
    plt. show()

def temp():
    x = []
    width = []
    height = []
    for img_dir in tqdm(glob('images/train/*.png')):
        img = cv2.imread (img_dir)
        height.append(img.shape[0])
        width.append(img.shape[1])

    print(min(height), min(width))



if __name__ == '__main__':
    # train()
    get_data()
    # temp()