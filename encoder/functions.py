import numpy as np
import math
import cv2
from typing import List , Tuple

import pickle
from scipy import fft
def crop_bitmap(bitmap: np.ndarray, size: int = 8) -> np.ndarray:
    return bitmap[math.floor(bitmap.shape[0] % size / 2):bitmap.shape[0] -
                  math.ceil(bitmap.shape[0] % 8 / 2),
                  math.floor(bitmap.shape[1] % size / 2):bitmap.shape[1] -
                  math.ceil(bitmap.shape[1] % size / 2), ]


def bgr_to_ycrcb(matrix3d: np.ndarray) -> np.ndarray:
    return (cv2.cvtColor(matrix3d, cv2.COLOR_BGR2YCrCb))

def __get_frequencies(matrix: np.ndarray, elements_count: int) -> list:
    return [c / elements_count for c in np.unique(matrix, return_counts=True)[1]]


def __get_entropy(freqs: list) -> float:
    return -sum(p * math.log2(p) for p in freqs)


def entropy(matrix: np.ndarray) -> float:
    return __get_entropy(__get_frequencies(matrix, np.prod(matrix.shape)))

def chromasubsample(matrix: np.ndarray):
    return matrix[::2, ::2]
def shape_for_contacting(shape: Tuple, size=8) -> Tuple:
    return math.ceil(shape[0] / size), math.ceil(shape[1] / size)

def split_to_ycbcr(matrix: np.ndarray) -> List[np.ndarray]:
    return [matrix[..., 0], matrix[..., 1], matrix[..., 2]]

def split_to_blocks(matrix: np.ndarray,
                                   size: int = 8) -> List[np.ndarray]:
    return [
        np.array([
            [
                matrix[row_index][col_index]
                for col_index in range(col, min(col + size, len(matrix[0])))
            ]  # row in matrix
            for row_index in range(row, min(row + size, len(matrix)))
        ])  # 8*8 matrix
        for row in range(0, len(matrix), size)
        for col in range(0, len(matrix[0]), size)
    ]


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





def quantization(submatrix: np.ndarray) -> np.ndarray:
    return np.array([[round(submatrix[row, col] / quantization_matrices[submatrix.shape[0]][row, col])
                      if quantization_matrices[submatrix.shape[0]][row, col] != 0
                      else 0
                      for col in range(len(submatrix[row]))]
                     for row in range(len(submatrix))])

def dct2(matrix: np.ndarray) -> np.ndarray:
    return(fft.dct(matrix))



def encode_image(src_path, dest_path) -> bool:
    print("read the file")
    bitmap =  cv2.imread(src_path)
    
    size=8
    

   
    bitmap = crop_bitmap(bitmap)

    print("convert RGB to YCrCb")
    bitmap = bgr_to_ycrcb(bitmap)
    #print(len(bitmap))
    #print(bitmap)
    bp=np.array(bitmap)
    #print(bp.shape)
    print("seperate bitmap to Y, Cb, Cr")
    y, cr, cb = split_to_ycbcr(bitmap)

    print("downsample")
    cr = chromasubsample(cr)
    cb = chromasubsample(cb)

    y_shape = shape_for_contacting(y.shape, size)
    cr_shape = shape_for_contacting(cr.shape, size)
    cb_shape = shape_for_contacting(cb.shape, size)
    # print(y_shape, y.shape)
    # print(cr_shape, cr.shape)
    # print(cb_shape, cb.shape)
    shapedata=[y_shape,cr_shape,cb_shape,bp.shape]
    file = open('shapedata16', 'wb')
    pickle.dump(shapedata, file)
    file.close()



    print("Splitting to {0}x{0} blocks".format(size))
    y = split_to_blocks(y, size)
    cr = split_to_blocks(cr, size)
    cb = split_to_blocks(cb, size)

    print("dct")
    y = [dct2(sub_matrix) for sub_matrix in y]
    cr = [dct2(sub_matrix) for sub_matrix in cr]
    cb = [dct2(sub_matrix) for sub_matrix in cb]

    print("Quantization")
    y = [quantization(submatrix) for submatrix in y]
    cr = [quantization(submatrix) for submatrix in cr]
    cb = [quantization(submatrix) for submatrix in cb]

    np.savez_compressed('imgdata16.npz', y_=y,cr_=cr,cb_=cb)
    

   
