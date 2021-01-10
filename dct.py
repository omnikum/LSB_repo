from __future__ import print_function
import cv2
import numpy as np
import itertools


quant = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 14, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])


#  Input: message - message message to be hidden
#          dest - name of the image you want to be output
#   Function: takes message to be hidden and preforms dct stegonography to hide the image within the least
#              significant bits of the DC coefficents.
#   Output: writes out an image with the encoded message
def encode_dct(src, message, dest):
    # load image for processing
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)

    if img is None:
        print("Error: File not found!")
        return 0

    message = str(len(message)) + '*' + message
    bit_mess = to_bits(src, message)

    # get size of image in pixels
    row, col = img.shape[:2]
    ori_row, ori_col = row, col

    if ((col / 8) * (row / 8) < len(message)):
        print("Error: Message too large to encode in image")
        return

    # make divisible by 8x8
    if row % 8 != 0 or col % 8 != 0:
        img = add_padd(img, row, col)

    row, col = img.shape[:2]

    # split image into RGB channels
    b_img, g_img, r_img = cv2.split(img)

    # message to be hid in blue channel so converted to type float32 for dct function
    b_img = np.float32(b_img)
    # print(b_img[0:8,0:8])

    # break into 8x8 blocks
    img_blocks = [np.round(b_img[j:j + 8, i:i + 8] - 128) for (j, i) in itertools.product(range(0, row, 8),
                                                                                        range(0, col, 8))]
    # print(img_blocks[1][0])
    # Blocks are run through DCT function
    dct_blocks = [np.round(cv2.dct(img_block)) for img_block in img_blocks]

    # blocks then run through quantization table
    quantized_dct = [np.round(dct_Block / quant) for dct_Block in dct_blocks]

    # set LSB in dc value corresponding bit of message
    mess_index = 0
    letter_index = 0

    for quantizedBlock in quantized_dct:
        # find LSB in dc coeff and replace with message bit
        dc = np.unpackbits(np.uint8(quantizedBlock[0][0]))

        dc[7] = bit_mess[mess_index][letter_index]

        dc = np.float32(np.packbits(dc)) - 255

        quantizedBlock[0][0] = dc

        letter_index = letter_index + 1
        if letter_index == 8:
            letter_index = 0
            mess_index = mess_index + 1
            if mess_index == len(message):
                break

    # blocks run inversely through quantization table
    s_img_blocks = [quantizedBlock * quant + 128 for quantizedBlock in quantized_dct]

    # blocks run through inverse DCT
    # s_img_blocks = [cv2.idct(B)+128 for B in quantized_dct]

    # puts the new image back together
    s_img = []
    for chunkRowBlocks in chunks(s_img_blocks, col / 8):
        for rowBlockNum in range(8):
            for block in chunkRowBlocks:
                s_img.extend(block[rowBlockNum])
    s_img = np.array(s_img).reshape(row, col)

    # converted from type float32
    s_img = np.uint8(s_img)

    s_img = cv2.merge((s_img, g_img, r_img))
    cv2.imwrite(dest, s_img)
    return s_img


#  Input: no input needed for function
#  Function: takes an image with a hidden dct encoded message and extracts the message into plaintext
#  Output: returns the plaintext string of the hidden message found"""
def decode_dct(src):
    res = decode(src)
    print(res)


def decode(src):
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)

    row, col = img.shape[:2]

    mess_size = None
    message_bits = []
    buff = 0

    # split image into RGB channels
    b_img, g_img, r_img = cv2.split(img)
    # print(b_img[0:8,0:8])
    # message hid in blue channel so converted to type float32 for dct function
    b_img = np.float32(b_img)
    # print(b_img[0:8,0:8])

    # break into 8x8 blocks
    img_blocks = [b_img[j:j + 8, i:i + 8] - 128 for (j, i) in itertools.product(range(0, row, 8),
                                                                              range(0, col, 8))]
    # blocks run through quantization table
    # quantized_dct = [dct_Block/ (quant) for dct_Block in dctBlocks]
    quantized_dct = [img_block / quant for img_block in img_blocks]
    # print(quantized_dct[1][0])
    i = 0
    # message extracted from LSB of DC coeff
    for quantizedBlock in quantized_dct:
        dc = np.unpackbits(np.uint8(quantizedBlock[0][0]))
        if dc[7] == 1:
            buff += (0 & 1) << (7 - i)
        elif dc[7] == 0:
            buff += (1 & 1) << (7 - i)
        i = 1 + i
        if i == 8:
            message_bits.append(chr(buff))
            buff = 0
            i = 0
            if message_bits[-1] == '*' and mess_size is None:
                try:
                    mess_size = int(''.join(message_bits[:-1]))
                except:
                    pass
        if len(message_bits) - len(str(mess_size)) - 1 == mess_size:
            return ''.join(message_bits)[len(str(mess_size)) + 1:]
    return 'error'


#  Helper function to 'stitch' new image back together
def chunks(l, n):
    m = int(n)
    for i in range(0, len(l), m):
        yield l[i:i + m]



#  Input: img-the image to be padded
#          row-the number of rows of pixels in the image
#          col-the number of columns in the image
#   Function: add 'Padding' making image dividable by 8x8 blocks
#   Output: returns the new padded image
def add_padd(img, row, col):
    img = cv2.resize(img, (col + (8 - col % 8), row + (8 - row % 8)))
    return img


#  Input: no inputs
#  Function: transforms the message that is wanted to be hidden from plaintext to a list of bits
#  Output: returns the list of strings of bits"""
def to_bits(src, message):
    bits = []
    for char in message:
        binval = bin(ord(char))[2:].rjust(8, '0')
        # for bit in binval:
        bits.append(binval)
    return bits