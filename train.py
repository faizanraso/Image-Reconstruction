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
    # model.compile(optimizer='adam', loss='mse')
    return model

def get_data():
    x = []
    y = []
    for img_dir in tqdm(glob('images/train/*.png')):
        img = cv2. imread (img_dir)
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y_channel = img_ycrcb[:,:, 0]
        y_out = cv2.resize(y_channel, (256, 256), interpolation=cv2.INTER_AREA)
        y_in = cv2.resize (y_out, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        x.append(y_in)
        y.append (y_out)
    
    x = np.array (x)
    y = np.array (y)

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

if __name__ == '__main__':
   train()