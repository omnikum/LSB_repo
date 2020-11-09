#import libraries
import sys
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

import random

CHUNK_LENGTH = 8  # длина блока кодирования

assert not CHUNK_LENGTH % 8, 'Длина блока должна быть кратна 8'  # проверка кратности длины блока кодирования

CHECK_BITS = [i for i in range(1, CHUNK_LENGTH + 1) if not i & (i - 1)]  # считаем контрольные биты

### HAMMING PART ###
def chars_to_bin(chars):
    """
    Преобразование символов в бинарный формат
    """
    assert not len(chars) * 8 % CHUNK_LENGTH, 'Длина кодируемых данных должна быть кратна длине блока кодирования'
    return ''.join([bin(ord(c))[2:].zfill(8) for c in chars])


def chunk_iterator(text_bin, chunk_size=CHUNK_LENGTH):
    """
    Поблочный вывод бинарных данных
    """
    for i in range(len(text_bin)):
        if not i % chunk_size:
            yield text_bin[i:i + chunk_size]


def get_check_bits_data(value_bin):
    """
    Получение информации о контрольных битах из бинарного блока данных
    """
    check_bits_count_map = {k: 0 for k in CHECK_BITS}
    for index, value in enumerate(value_bin, 1):
        if int(value):
            bin_char_list = list(bin(index)[2:].zfill(8))
            bin_char_list.reverse()
            for degree in [2 ** int(i) for i, value in enumerate(bin_char_list) if int(value)]:
                check_bits_count_map[degree] += 1
    check_bits_value_map = {}
    for check_bit, count in check_bits_count_map.items():
        check_bits_value_map[check_bit] = 0 if not count % 2 else 1
    return check_bits_value_map


def set_empty_check_bits(value_bin):
    """
    Добавить в бинарный блок "пустые" контрольные биты
    """
    for bit in CHECK_BITS:
        value_bin = value_bin[:bit - 1] + '0' + value_bin[bit - 1:]
    return value_bin


def set_check_bits(value_bin):
    """
    Установить значения контрольных бит
    """
    value_bin = set_empty_check_bits(value_bin)
    check_bits_data = get_check_bits_data(value_bin)
    for check_bit, bit_value in check_bits_data.items():
        value_bin = '{0}{1}{2}'.format(
            value_bin[:check_bit - 1], bit_value, value_bin[check_bit:])
    return value_bin


def get_check_bits(value_bin):
    """
    Получить информацию о контрольных битах из блока бинарных данных
    """
    check_bits = {}
    for index, value in enumerate(value_bin, 1):
        if index in CHECK_BITS:
            check_bits[index] = int(value)
    return check_bits


def exclude_check_bits(value_bin):
    """
    Исключить информацию о контрольных битах из блока бинарных данных
    """
    clean_value_bin = ''
    for index, char_bin in enumerate(list(value_bin), 1):
        if index not in CHECK_BITS:
            clean_value_bin += char_bin

    return clean_value_bin


def set_errors(encoded):
    """
    Допустить ошибку в блоках бинарных данных
    """
    result = ''
    for chunk in chunk_iterator(encoded, CHUNK_LENGTH + len(CHECK_BITS)):
        num_bit = random.randint(1, len(chunk))
        chunk = '{0}{1}{2}'.format(chunk[:num_bit - 1], int(chunk[num_bit - 1]) ^ 1, chunk[num_bit:])
        result += (chunk)
    return result


def check_and_fix_error(encoded_chunk):
    """
    Проверка и исправление ошибки в блоке бинарных данных
    """
    check_bits_encoded = get_check_bits(encoded_chunk)
    check_item = exclude_check_bits(encoded_chunk)
    check_item = set_check_bits(check_item)
    check_bits = get_check_bits(check_item)
    if check_bits_encoded != check_bits:
        invalid_bits = []
        for check_bit_encoded, value in check_bits_encoded.items():
            if check_bits[check_bit_encoded] != value:
                invalid_bits.append(check_bit_encoded)
        num_bit = sum(invalid_bits)
        encoded_chunk = '{0}{1}{2}'.format(
            encoded_chunk[:num_bit - 1],
            int(encoded_chunk[num_bit - 1]) ^ 1,
            encoded_chunk[num_bit:])
    return encoded_chunk


def get_diff_index_list(value_bin1, value_bin2):
    """
    Получить список индексов различающихся битов
    """
    diff_index_list = []
    for index, char_bin_items in enumerate(zip(list(value_bin1), list(value_bin2)), 1):
        if char_bin_items[0] != char_bin_items[1]:
            diff_index_list.append(index)
    return diff_index_list


def decode_hamming(encoded, fix_errors=True):
    """
    Декодирование данных
    """
    decoded_value = ''
    fixed_encoded_list = []
    for encoded_chunk in chunk_iterator(encoded, CHUNK_LENGTH + len(CHECK_BITS)):
        if fix_errors:
            encoded_chunk = check_and_fix_error(encoded_chunk)
        fixed_encoded_list.append(encoded_chunk)

    clean_chunk_list = []
    for encoded_chunk in fixed_encoded_list:
        encoded_chunk = exclude_check_bits(encoded_chunk)
        clean_chunk_list.append(encoded_chunk)

    for clean_chunk in clean_chunk_list:
        for clean_char in [clean_chunk[i:i + 8] for i in range(len(clean_chunk)) if not i % 8]:
            decoded_value += chr(int(clean_char, 2))
    return decoded_value


def encode_hamming(message):

    assert not len(message) * 8 % CHUNK_LENGTH, 'Длина кодируемых данных должна быть кратна длине блока кодирования'
    text_bin = ''.join([bin(ord(c))[2:].zfill(8) for c in message])
    result = ''

    for chunk_bin in chunk_iterator(text_bin):
        chunk_bin = set_check_bits(chunk_bin)
        result += chunk_bin
    print("Encoded message is: ", result)
    return result


### IMAGE PART ###
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
    if "EOF" in message:
        print("Hidden Message:", message[:-3])
    elif hamming:
        message = decode_hamming(message[:-3])
        print("Encoded Message:", message)
    else:
        print("No Hidden Message Found")


### PSNR PART ###
def calculate_psnr(img1, img2, img3):

    r_img1 = cv2.imread(img1)
    r_img2 = cv2.imread(img2)
    r_img3 = cv2.imread(img3)
    psnr12 = cv2.PSNR(r_img1, r_img2)
    psnr13 = cv2.PSNR(r_img1, r_img3)
    psnr23 = cv2.PSNR(r_img2, r_img3)
    print("PSNR 1-2 is: ", psnr12)
    print("PSNR 1-3 is: ", psnr13)
    print("PSNR 2-3 is: ", psnr23)

### HISTOGRAMM PART ###
def create_hist(img1, img2, img3):

    r_img1 = cv2.imread(img1)
    vals1 = r_img1.mean(axis=2).flatten()
    counts1, bins1 = np.histogram(vals1, range(257))
    r_img2 = cv2.imread(img2)
    vals2 = r_img2.mean(axis=2).flatten()
    counts2, bins2 = np.histogram(vals2, range(257))
    r_img3 = cv2.imread(img3)
    vals3 = r_img3.mean(axis=2).flatten()
    counts3, bins3 = np.histogram(vals3, range(257))
    plt.hist([counts1, counts2, counts3], bins1[:-1] - 0.5, histtype='bar')
    plt.xlim([-0.5, 255])
    plt.show()


### BIT PLANE PART ###
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

### MAIN PART ###
def main_stego():

    print("1: Encode")
    print("2: Decode")
    print("3: Encode using Hamming code")
    print("4: Decode image with Hamming code")
    print("5: Calculate PSNR of two images")
    print("6: Create Histograms")
    print("7: Create bit plane images")
    menu = input()

    if menu == '1':
        print("Input Source Image Path")
        src = input()
        print("Input Message to Hide")
        message = input()
        print("Input Destination Image Path")
        dest = input()
        print("Encoding...")
        encode_image(src, message, dest)

    elif menu == '2':
        print("Input Source Image Path")
        src = input()
        print("Decoding...")
        decode_image(src)

    elif menu == '3':
        print("Input Source Image Path")
        src = input()
        print("Input Message to Hide")
        message = input()
        print("Input Destination Image Path")
        dest = input()
        print("Encoding with Hamming code...")
        encode_image(src, message, dest, hamming=True)

    elif menu == '4':
        print("Input Source Image Path")
        src = input()
        print("Decoding image with Hamming code...")
        decode_image(src, hamming=True)

    elif menu == '5':
        print("Calculate PSNR")
        print("Input Source Path Of The First Image")
        src1 = input()
        print("Input Source Path Of The Second Image")
        src2 = input()
        print("Input Source Path Of The Third Image")
        src3 = input()
        calculate_psnr(src1, src2, src3)

    elif menu == '6':
        print("Create histograms")
        print("Input Source Path Of The First Image")
        src1 = input()
        print("Input Source Path Of The Second Image")
        src2 = input()
        print("Input Source Path Of The Third Image")
        src3 = input()
        create_hist(src1, src2, src3)

    elif menu == '7':
        print("Create bit plane of image")
        print("Input Source Path Of The Image")
        src1 = input()
        create_bit_plane(src1)

    else:
        print("ERROR: Invalid option chosen")


main_stego()
