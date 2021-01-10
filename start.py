from lsb import encode_image, decode_image
from analyze import calculate_psnr, create_hist, create_bit_plane, chi_squared_test
from dct import decode_dct, encode_dct

if __name__ == "__main__":
    print("1: Encode")
    print("2: Decode")
    print("3: Encode using Hamming code")
    print("4: Decode image with Hamming code")
    print("5: Calculate PSNR of two images")
    print("6: Create Histograms")
    print("7: Create bit plane images")
    print("8: Chi squared test")
    print("9: Encode jpeg (DCT)")
    print("10: Decode jpeg (DCT)")
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
        calculate_psnr(src1, src2)

    elif menu == '6':
        print("Create histograms")
        print("Input Source Path Of The First Image")
        src1 = input()
        print("Input Source Path Of The Second Image")
        src2 = input()
        create_hist(src1, src2)

    elif menu == '7':
        print("Create bit plane of image")
        print("Input Source Path Of The Image")
        src1 = input()
        create_bit_plane(src1)

    elif menu == '8':
        print("Chi squared test")
        print("Input Source Path Of The Image")
        src1 = input()
        chi_squared_test(src1)

    elif menu == '9':
        print("Input Source Image Path")
        src = input()
        print("Input Message to Hide")
        message = input()
        print("Input Destination Image Path")
        dest = input()
        print("Encoding...")
        encode_dct(src, message, dest)

    elif menu == '10':
        print("Input Source Image Path")
        src = input()
        print("Decoding...")
        decode_dct(src)

    else:
        print("ERROR: Invalid option chosen")
