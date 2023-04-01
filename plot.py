import matplotlib.pyplot as plt
import cv2

# This function is used to plot the Y, U and V channels in one plot

def plot():
    img_number = "5"
    y_image =  cv2.imread("./images/yuv_images/img" + img_number +"/y_channel.jpeg")
    u_image =  cv2.imread("./images/yuv_images/img" + img_number +"/u_channel.jpeg")
    v_image =  cv2.imread("./images/yuv_images/img" + img_number +"/v_channel.jpeg")

    fig = plt.figure(figsize=(2, 7))
    
    fig.add_subplot(3, 1, 1)
    plt.imshow(y_image)
    plt.axis('on')
    plt.title("Y Channel", fontdict={'fontsize': 10})
    
    fig.add_subplot(3, 1, 2)
    plt.imshow(u_image)
    plt.axis('on')
    plt.title("U Channel", fontdict={'fontsize': 10})

    fig.add_subplot(3, 1, 3)
    plt.imshow(v_image)
    plt.axis('on')
    plt.title("V Channel", fontdict={'fontsize': 10})

    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)

    plt.show()
    

if __name__ == '__main__':
    plot()