import cv2
import numpy as np
import math
from tensorflow import keras
from keras.layers import *
from keras.models import *
import cv2
import numpy as np 
import matplotlib.pyplot as plt

def main():
    img_number = "1"
    image = cv2.imread("./images/input/img" + img_number + ".jpg", 1)
    B, G, R = cv2.split(image)

    if(image.shape[0] % 2 != 0):
        image = image[:-1, :, :]
    if(image.shape[1] % 2 != 0):
        image = image[:, :-1, :]

    Y, U, V = rbg_to_yuv(R, G, B)
    Y, U, V = floor_values(Y, U, V)
    Y, U, V = downsample(Y, U, V)

    Y = upsample(Y, 'model_y')
    U = upsample(U, 'model_u')
    V = upsample(V, 'model_v')

    Y, U, V = floor_values(Y, U, V)
    R, G, B = yuv_to_rgb(Y, U, V)

    new_image = cv2.merge([B, G, R]).astype(np.uint8)

    cv2.imwrite("./images/output/img" + img_number + "_out.png", new_image)
    PSNR = calculate_psnr(image, new_image)
    SSIM = calculate_ssim(image, new_image)

    print(f"PSNR: {PSNR} dB")
    print(f"SSIM: {SSIM}")

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

# billinear upsampling
def upsample(channel, model_name):
    model = load_model('./data/models/' + model_name + '.h5')
    channel = np.expand_dims(channel, axis=0)
    channel_upsampled = model.predict(channel)
    channel_upsampled = (np.asarray(np.floor(channel_upsampled), dtype='int'))
    
    return channel_upsampled[0]

# convert from YUV to RGB
def yuv_to_rgb(y, u, v):
    r = np.zeros(y.shape, dtype=np.float32)
    g = np.zeros(y.shape, dtype=np.float32)
    b = np.zeros(y.shape, dtype=np.float32)

    r = y + 1.4075 * (v - 128)
    g = y - 0.3455 * (u - 128) - (0.7169 * (v - 128))
    b = y + 1.7790 * (u - 128)

    return np.clip(r, 0, 255).astype(np.uint8), np.clip(g, 0, 255).astype(np.uint8), np.clip(b, 0, 255).astype(np.uint8)

# calculate PSNR
def calculate_psnr(original_image, new_image):
    mean_squared_error = np.mean((original_image - new_image) ** 2)
    if (mean_squared_error != 0):
        ratio = 20 * math.log10(255 / math.sqrt(mean_squared_error))
        return ratio
    else:
        return math.inf

# calculate SSIM
def calculate_ssim(img_1, img_2):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    img_1 = img_1.astype(np.double)
    img_2 = img_2.astype(np.double)
    kernel = (11, 11)
    sigma = 1.5

    mu_x = cv2.GaussianBlur(img_1, kernel, sigma)
    mu_y = cv2.GaussianBlur(img_2, kernel, sigma)
    var_x = cv2.GaussianBlur(img_1 ** 2, kernel, sigma) - (mu_x ** 2)
    var_y = cv2.GaussianBlur(img_2 ** 2, kernel, sigma) - (mu_y ** 2)
    covar_xy = cv2.GaussianBlur(img_1 * img_2, kernel, sigma) - (mu_x * mu_y)

    ssim = ((2 * (mu_x * mu_y) + c1) * (2 * covar_xy + c2)) / (((mu_x ** 2) + (mu_y ** 2) + c1) * (var_x + var_y + c2))
    return ssim.mean()


if __name__ == "__main__":
    main()