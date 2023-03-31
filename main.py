import cv2
import numpy as np
import math


def main():
    image = cv2.imread("./image.jpeg", 1)
    B, G, R = cv2.split(image)

    # Make sure the image is even - this will aid in upsampling later
    if(image.shape[0] % 2 != 0):
        image = image[:-1, :, :]
    if(image.shape[1] % 2 != 0):
        image = image[:, :-1, :]


    Y, U, V = rbg_to_yuv(R, G, B)
    Y, U, V = floor_values(Y, U, V)
    Y, U, V = downsample(Y, U, V)

    # Save copies of the YUV channels
    # cv2.imwrite("y_channel.jpeg", Y.astype(np.uint8))
    # cv2.imwrite("u_channel.jpeg", U.astype(np.uint8))
    # cv2.imwrite("v_channel.jpeg", V.astype(np.uint8))

    # make sure the size of all components are the same
    new_height, new_width = int(Y.shape[0]*2), int(Y.shape[1])*2

    Y = upsample(Y, new_height, new_width, 2)
    U = upsample(U, new_height, new_width, 4)
    V = upsample(V, new_height, new_width, 4)

    Y, U, V = floor_values(Y, U, V)
    R, G, B = (yuv_to_rgb(Y, U, V))

    new_image = cv2.merge([B, G, R]).astype(np.uint8)

    cv2.imwrite("new_image.jpeg", new_image)
    value = calculate_psnr(image, new_image)
    print(f"PSNR: {value} dB")

    cv2.imshow("Image", new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
    
    pass



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


if __name__ == "__main__":
    main()
