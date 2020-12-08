import numpy as np
from PIL import Image
from hamming import encode_hamming, decode_hamming


def encode_image(src, message, dest, hamming=False):

    if hamming:
        message = encode_hamming(message)

    img = Image.open(src, 'r')
    width, height = img.size
    array = np.array(list(img.getdata()))

    n = 3
    m = 0

    if img.mode == 'RGBA':
        n = 4
        m = 1

    total_pixels = array.size//n

    message += "EOM"
    b_message = ''.join([format(ord(i), "08b") for i in message])
    req_pixels = len(b_message)

    if req_pixels > total_pixels:
        print("ERROR: Need larger file size")

    else:
        index = 0
        for p in range(total_pixels):
            for q in range(m, n):
                if index < req_pixels:
                    array[p][q] = int(bin(array[p][q])[2:9] + b_message[index], 2)
                    index += 1

        array = array.reshape(height, width, n)
        enc_img = Image.fromarray(array.astype('uint8'), img.mode)
        enc_img.save(dest)
        print("Image Encoded Successfully")


# decoding function
def decode_image(src, hamming=False):

    img = Image.open(src, 'r')
    array = np.array(list(img.getdata()))

    n = 3
    m = 0

    if img.mode == 'RGBA':
        n = 4
        m = 1

    total_pixels = array.size//n

    hidden_bits = ""
    for p in range(total_pixels):
        for q in range(m, n):
            hidden_bits += (bin(array[p][q])[2:][-1])

    hidden_bits = [hidden_bits[i:i+8] for i in range(0, len(hidden_bits), 8)]

    message = ""
    for i in range(len(hidden_bits)):
        if message[-3:] == "EOM":
            break
        else:
            message += chr(int(hidden_bits[i], 2))
    if "EOM" in message:
        print("Hidden Message:", message[:-3])
        if hamming:
            message = decode_hamming(message[:-3])
            print("Encoded Message:", message)
    else:
        print("No Hidden Message Found")