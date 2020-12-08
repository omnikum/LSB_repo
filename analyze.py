import cv2
import numpy as np
import matplotlib.pyplot as plt


# PSNR PART
def calculate_psnr(img1=None, img2=None):

    r_img1 = cv2.imread(img1)
    r_img2 = cv2.imread(img2)
    psnr = cv2.PSNR(r_img1, r_img2)
    print("PSNR {1}-{2} is: ".format(img1, img2), psnr)


# HISTOGRAMM PART
def create_hist(img1, img2):

    r_img1 = cv2.imread(img1)
    vals1 = r_img1.mean(axis=2).flatten()
    counts1, bins1 = np.histogram(vals1, range(257))
    r_img2 = cv2.imread(img2)
    vals2 = r_img2.mean(axis=2).flatten()
    counts2, bins2 = np.histogram(vals2, range(257))
    plt.hist([counts1, counts2], bins1[:-1] - 0.5, histtype='bar')
    plt.xlim([-0.5, 255])
    plt.show()


# BIT PLANE PART
def create_bit_plane(image):

    # Read the image in greyscale
    img = cv2.imread(image, 0)
    if img is None:
        print("you fucked up")

    # Iterate over each pixel and change pixel value to binary using np.binary_repr() and store it in a list.
    lst = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            lst.append(np.binary_repr(img[i][j], width=8))  # width = no. of bits

    # We have a list of strings where each string represents binary pixel value.
    # To extract bit planes we need to iterate over the strings and store the
    # characters corresponding to bit planes into lists.
    # Multiply with 2^(n-1) and reshape to reconstruct the bit image.
    eight_bit_img = (np.array([int(i[0]) for i in lst], dtype=np.uint8) * 128).reshape(img.shape[0], img.shape[1])
    print("8bit")
    seven_bit_img = (np.array([int(i[1]) for i in lst], dtype=np.uint8) * 64).reshape(img.shape[0], img.shape[1])
    print("7bit")
    six_bit_img = (np.array([int(i[2]) for i in lst], dtype=np.uint8) * 32).reshape(img.shape[0], img.shape[1])
    print("6bit")
    five_bit_img = (np.array([int(i[3]) for i in lst], dtype=np.uint8) * 16).reshape(img.shape[0], img.shape[1])
    print("5bit")
    four_bit_img = (np.array([int(i[4]) for i in lst], dtype=np.uint8) * 8).reshape(img.shape[0], img.shape[1])
    print("4bit")
    three_bit_img = (np.array([int(i[5]) for i in lst], dtype=np.uint8) * 4).reshape(img.shape[0], img.shape[1])
    print("3bit")
    two_bit_img = (np.array([int(i[6]) for i in lst], dtype=np.uint8) * 2).reshape(img.shape[0], img.shape[1])
    print("2bit")
    one_bit_img = (np.array([int(i[7]) for i in lst], dtype=np.uint8) * 1).reshape(img.shape[0], img.shape[1])
    print("1bit")

    # Concatenate these images for ease of display using cv2.hconcat()
    finalr = cv2.hconcat([eight_bit_img, seven_bit_img, six_bit_img, five_bit_img])
    finalv = cv2.hconcat([four_bit_img, three_bit_img, two_bit_img, one_bit_img])

    # Vertically concatenate
    final = cv2.vconcat([finalr, finalv])

    # Display the images
    plt.imshow(final)
    plt.show()