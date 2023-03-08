import numpy as np
import math
from typing import Tuple, List
import cv2
from scipy import fft
import pickle

quantization_matrices = {
    8: np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
}
quantization_matrices[16] = quantization_matrices[8].repeat(2, axis=0).repeat(2, axis=1)
quantization_matrices[32] = quantization_matrices[16].repeat(2, axis=0).repeat(2, axis=1)
def un_quantization(matrix: np.ndarray) -> np.ndarray:
    return np.array([[matrix[row, col] * quantization_matrices[matrix.shape[0]][row, col]
                      for col in range(len(matrix[row]))]
                     for row in range(len(matrix))])

def concatenate_sub_matrices_to_big_matrix(submatrices: List[np.ndarray],
                                           shape: Tuple[int, int]):
    return np.block([
        submatrices[i:i + shape[1]]
        for i in range(0, len(submatrices), shape[1])
    ])


def concatenate_three_colors(y: np.ndarray, cr: np.ndarray,
                             cb: np.ndarray, out: np.ndarray = None) -> np.ndarray:
    return np.stack((y, cr, cb), 2, out)

def upsample(matrix: np.ndarray) -> np.ndarray:
    return matrix.repeat(2, axis=0).repeat(2, axis=1)


def ycrcb_pixel_to_bgr(ycrcb: list) -> List[np.uint8]:
    return [
        np.uint8(np.clip(round(ycrcb[0] + 1.773 * (ycrcb[2] - 128)), 0, 255)),  # B
        np.uint8(np.clip(round(ycrcb[0] - 0.714 * (ycrcb[1] - 128) - 0.344 * (ycrcb[2] - 128)), 0, 255)),  # G
        np.uint8(np.clip(round(ycrcb[0] + 1.403 * (ycrcb[1] - 128)), 0, 255))  # R
    ]




def idct2(matrix: np.ndarray) -> np.ndarray:
    return(fft.idct(matrix))
def ycrcb_to_bgr(matrix3d: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(matrix3d, cv2.COLOR_YCR_CB2BGR)

def save_img(matrix: np.ndarray, dest: str, mode: str = 'RGB'):
    if mode == 'YCrCb':
        matrix = ycrcb_to_bgr(matrix)
    cv2.imwrite(dest, matrix)
def decode_image(src_path, dest_path):
    #load data

    print("reading file")
    
    loaded = np.load(src_path)
   
    print()
    y=loaded['y_']
    cr=loaded['cr_']
    cb=loaded['cb_']
    
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
    y = [idct2(matrix) for matrix in y]
    cr = [idct2(matrix) for matrix in cr]
    cb = [idct2(matrix) for matrix in cb]

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
    bitmap=np.zeros((bp_shape),dtype='uint8')
    y1 = y.astype('uint8')
    cb1 = cb.astype('uint8')
    cr1 = cr.astype('uint8')
    print("concatenate YCbCr into one array")
    concatenate_three_colors(y1, cr1, cb1, bitmap)

    # print("ycrcb_to_bgr")
    # bitmap = imagetools.ycrcb_to_bgr(bitmap)

    print("save img")
    save_img(bitmap, dest_path,mode='YCrCb')


