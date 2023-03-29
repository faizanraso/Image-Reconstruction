from tensorflow import keras
from keras.layers import *
from keras.models import *
from glob import glob 
import cv2
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

def get_data_y():
    x = []
    y = []
    for img_dir in tqdm(glob('images/train/*.png')):
        img = cv2. imread (img_dir)
        B,G,R = cv2.split()
        Y, U, V = rbg_to_yuv(R, G, B)
        Y, U, V = floor_values(Y, U, V)

        y_channel = Y
        y_out = y_channel
        
        Y, U, V = downsample(Y, U, V)
        y_in = Y

        x.append(y_in)
        y.append (y_out)
    
    x = np.array (x)
    y = np.array (y)

    return x, y

def get_data_u():
    x = []
    y = []
    for img_dir in tqdm(glob('images/train/*.png')):
        img = cv2. imread (img_dir)
        B,G,R = cv2.split()
        Y, U, V = rbg_to_yuv(R, G, B)
        Y, U, V = floor_values(Y, U, V)

        u_channel = U
        y_out = u_channel
        
        Y, U, V = downsample(Y, U, V)
        y_in = Y

        x.append(y_in)
        y.append (y_out)
    
    x = np.array (x)
    y = np.array (y)

    return x, y

def get_data_v():
    x = []
    y = []
    for img_dir in tqdm(glob('images/train/*.png')):
        img = cv2. imread (img_dir)
        B,G,R = cv2.split()
        Y, U, V = rbg_to_yuv(R, G, B)
        Y, U, V = floor_values(Y, U, V)

        v_channel = V
        y_out = v_channel
        
        Y, U, V = downsample(Y, U, V)
        y_in = Y

        x.append(y_in)
        y.append (y_out)
    
    x = np.array (x)
    y = np.array (y)

    return x, y

def train():
    model = get_model()
    y_x, y_y = get_data_y()
    print (y_x.shape, y_y.shape)

    y_x_train, y_x_val, y_y_train, y_y_val = train_test_split(y_x, y_y, test_size=0.2, random_state=42)
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
    model.fit(y_x_train, y_y_train, batch_size=batch_size, epochs=epochs, validation_data=(y_x_val, y_y_val), validation_split=0.1, callbacks=[save_model_callback, tb_callback])


def run_model():
    model = load_model('model/model.h5')
    img = cv2.imread('images/valid/0801.png')
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_channel = img_ycrcb[:, :, 0]
    # y_in = cv2.resize(y_channel, (256, 256), interpolation=cv2.INTER_AREA)
    y = np.expand_dims(y_channel, axis=0)
    y_upsampled = model.predict(y)
    
    plt.subplot(211)
    plt.imshow(y[0], cmap='gray') 
    plt.subplot (212) 
    plt.imshow(y_upsampled[0], cmap='gray') 
    plt.show()

    return y_upsampled

if __name__ == '__main__':
#    train()
   run_model()