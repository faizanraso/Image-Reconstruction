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
    img_number = "5"
    image = cv2.imread("./images/input/img" + img_number + ".jpg", 1)
    B, G, R = cv2.split(image)

    if(image.shape[0] % 2 != 0):
        image = image[:-1, :, :]
    if(image.shape[1] % 2 != 0):
        image = image[:, :-1, :]

    Y, U, V = rbg_to_yuv(R, G, B)
    Y, U, V = floor_values(Y, U, V)
    Y, U, V = downsample(Y, U, V)

    # Save copies of the YUV channels
    cv2.imwrite("./images/yuv_images/img" + img_number + "/y_channel.jpeg", Y.astype(np.uint8))
    cv2.imwrite("./images/yuv_images/img" + img_number + "/u_channel.jpeg", U.astype(np.uint8))
    cv2.imwrite("./images/yuv_images/img" + img_number + "/v_channel.jpeg", V.astype(np.uint8))

    Y = upsample(Y, 'model_y')
    U = upsample(U, 'model_u')
    V = upsample(V, 'model_v')

    Y, U, V = floor_values(Y, U, V)
    R, G, B = yuv_to_rgb(Y, U, V)

    new_image = cv2.merge([B, G, R]).astype(np.uint8)

    cv2.imwrite("./images/output/img" + img_number + "_out.png", new_image)
    PSNR = calculate_psnr(image, new_image)
    SSIM = ssim(image, new_image)

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
def ssim(img_1, img_2):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    img_1 = img_1.astype(np.float64)
    img_2 = img_2.astype(np.float64)
    mu_1 = cv2.GaussianBlur(img_1, (11, 11), 1.5)
    mu_2 = cv2.GaussianBlur(img_2, (11, 11), 1.5)
    mu_1_sq = mu_1 ** 2
    mu_2_sq = mu_2 ** 2
    mu_1_mu_2 = mu_1 * mu_2
    sigma_1_sq = cv2.GaussianBlur(img_1 * img_1, (11, 11), 1.5) - mu_1_sq
    sigma_2_sq = cv2.GaussianBlur(img_2 * img_2, (11, 11), 1.5) - mu_2_sq
    sigma_12 = cv2.GaussianBlur(img_1 * img_2, (11, 11), 1.5) - mu_1_mu_2
    ssim_map = ((2 * mu_1_mu_2 + c1) * (2 * sigma_12 + c2)) / ((mu_1_sq + mu_2_sq + c1) * (sigma_1_sq + sigma_2_sq + c2))
    return ssim_map.mean()


if __name__ == "__main__":
    main()