from __future__ import print_function
import cv2
import numpy as np
import itertools

"""
TODO:

1. Оптимизация. Убрать округление (np.round) при считывании img_blocks и dct_blocks (в кодере и декодере)
2. Декодинг по четности элемента. Встраивать бит сообщения не в нулевой бит элемента матрицы ДКП, а в последний (всего 
8 бит, соответственно встраивать нужно в dc[7], декодер в конце программы, который закомменчен - можно использовать
 для декодинга по четности)
3.Увеличение размера сообщения. Встраивать бит сообщения не только в элемент [1][1] матрицы ДКП, а во все элементы 
матрицы ДКП, кроме [0][0] (требует дополнительных экспериментов)

Опционально:
1. Гонять сообщение через код Хэмминга - увеличится длина сообщения, но меньше вероятность ошибки при пересчете ДКП
"""



def chunks(s_img_blocks, col_na_8):
    shag = int(col_na_8)
    for i in range(0, len(s_img_blocks), shag):
        yield s_img_blocks[i:i + shag]


def add_padd(img, row, col):
    img = cv2.resize(img, (col + (8 - col % 8), row + (8 - row % 8)))
    return img


def to_bits(message):
    bits = []
    for char in message:
        binval = bin(ord(char))[2:].rjust(8, '0')
        # for bit in binval:
        bits.append(binval)
    return bits


def encode_dct(src, message, dest):
    # load image for processing
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)

    if img is None:
        print("Error: File not found!")
        return 0

    message = message + '*'
    bit_mess = to_bits(message)
    #print(bit_mess)
    # get size of image in pixels
    row, col = img.shape[:2]

    if (col / 8) * (row / 8) < len(message):
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

    # break into 8x8 blocks
    img_blocks = [np.round(b_img[j:j + 8, i:i + 8]) for (j, i) in itertools.product(range(0, row, 8), range(0, col, 8))]

    # Blocks are run through DCT function
    dct_blocks = [np.round(cv2.dct(img_block)) for img_block in img_blocks]

    # set LSB in dc value corresponding bit of message
    mess_index = 0
    letter_index = 0

    for dct_block in dct_blocks:

        # find LSB in dc coeff and replace with message bit
        dc = np.unpackbits(np.round(np.uint8(dct_block[1][1])))

        dc[0] = bit_mess[mess_index][letter_index]

        dc = np.float32(np.packbits(dc))

        dct_block[1][1] = dc

        letter_index = letter_index + 1
        if letter_index == 8:
            letter_index = 0
            mess_index = mess_index + 1
            if mess_index == len(message):
                break

    # blocks run through inverse DCT
    bl_img_blocks = [(cv2.idct(B)) for B in dct_blocks]

    # puts the new image back together
    bl_img = []
    for chunkRowBlocks in chunks(bl_img_blocks, col / 8):
        for rowBlockNum in range(8):
            for block in chunkRowBlocks:
                bl_img.extend(block[rowBlockNum])
    bl_img = np.array(bl_img).reshape(row, col)

    # converted from type float32
    bl_img = np.uint8(bl_img)

    bl_img = cv2.merge((bl_img, g_img, r_img))
    cv2.imwrite(dest, bl_img)

    print("Encoding done!")
    return bl_img

def decode_dct(src):
    print("Decoding...")
    res = decode(src)
    print(res)


def decode(src):
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)

    row, col = img.shape[:2]

    # split image into RGB channels
    b_img, g_img, r_img = cv2.split(img)

    # message hid in blue channel so converted to type float32 for dct function
    b_img = np.float32(b_img)
    # print(b_img[0:8,0:8])

    # break into 8x8 blocks
    img_blocks = [np.round(b_img[j:j + 8, i:i + 8]) for (j, i) in itertools.product(range(0, row, 8),
                                                                              range(0, col, 8))]

    dct_blocks = [np.round(cv2.dct(img_block)) for img_block in img_blocks]

    buffer = ''
    res_mes = ''
    for dct_block in dct_blocks:

        # find LSB in dc coeff and replace with message bit
        dc = np.unpackbits(np.round(np.uint8(dct_block[1][1])))  # поменять на [7][7]

        buffer += str(dc[0])
        if len(buffer) == 8:
            #print(buffer)
            if chr(int(buffer, 2)) != '*':
                res_mes += chr(int(buffer, 2))
                buffer = ''
            else:
                print(res_mes)
                break
                #return res_mes
#    message extracted from LSB of DC coeff
#    for dct_block in dct_blocks:
#        if dct_block[1][1] % 2 == 0:
#            buffer += '0'
#        else:
#            buffer += '1'
#        if len(buffer) == 8:
#            #print(buffer)
#            if chr(int(buffer, 2)) != '*':
#                res_mes += chr(int(buffer, 2))
#                buffer = ''
#            else:
#                print(res_mes)
#                #return res_mes
