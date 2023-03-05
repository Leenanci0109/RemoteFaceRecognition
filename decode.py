import pickle
import numpy as np
import sys
import argparse
from functions import *

def main():
    parser = argparse.ArgumentParser(
        description='Jpeg encoder')
    parser.add_argument(
        'SRC',
        help='path to original image'
    )
    parser.add_argument(
        'DST',
        help=
        'Path to directry to save file',
        default='./')
    # open a file, where you stored the pickled data
    file = open('sendfile', 'rb')

    # dump information to that file
    data = pickle.load(file)

    # close the file
    file.close()
    print()
    y=np.array(data[0])
    cb=np.array(data[1])
    cr=np.array(data[2])
    print(y.shape)
    print(cb.shape)
    print(cr.shape)
    print("UnQuantization")
    y = [
        un_quantization(submatrix) for submatrix in y
    ]
    cr = [
        un_quantization(submatrix) for submatrix in cr
    ]
    cb = [
        un_quantization(submatrix) for submatrix in cb
    ]

    print("Invert dct")
    y = [inverse_dct(matrix) for matrix in y]
    cr = [inverse_dct(matrix) for matrix in cr]
    cb = [inverse_dct(matrix) for matrix in cb]

    file = open('shapedata', 'rb')

    # dump information to that file
    data = pickle.load(file)
    y_shape=data[0]
    cr_shape=data[1]
    cb_shape=data[2]
    bp_shape=data[3]
    print(bp_shape)
    # close the file
    file.close()

    print("Concatenate")
    y = concatenate_sub_matrices_to_big_matrix(y, y_shape)
    cr = concatenate_sub_matrices_to_big_matrix(cr, cb_shape)
    cb = concatenate_sub_matrices_to_big_matrix(cb, cr_shape)

    print("upsample")
    cr = upsample(cr)
    cb = upsample(cb)
    bitmap=np.zeros((bp_shape))

    print("concatenate")
    concatenate_three_colors(y, cr, cb, bitmap)

    # print("ycrcb_to_bgr")
    # bitmap = imagetools.ycrcb_to_bgr(bitmap)

    print("save img")
    save_img(bitmap, mode='YCrCb')


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())