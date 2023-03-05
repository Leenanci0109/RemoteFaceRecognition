import numpy as np
import math
import cv2
from typing import List , Tuple
from functools import lru_cache
import pickle

def __get_frequencies(matrix: np.ndarray, elements_count: int) -> list:
    return [c / elements_count for c in np.unique(matrix, return_counts=True)[1]]


def __get_entropy(freqs: list) -> float:
    return -sum(p * math.log2(p) for p in freqs)


def entropy(matrix: np.ndarray) -> float:
    return __get_entropy(__get_frequencies(matrix, np.prod(matrix.shape)))
def crop_bitmap(bitmap: np.ndarray, size: int = 8) -> np.ndarray:
    return bitmap[math.floor(bitmap.shape[0] % size / 2):bitmap.shape[0] -
                  math.ceil(bitmap.shape[0] % 8 / 2),
                  math.floor(bitmap.shape[1] % size / 2):bitmap.shape[1] -
                  math.ceil(bitmap.shape[1] % size / 2), ]


def calculate_y(bgr: list) -> float:
    return 0.299 * bgr[2] + 0.587 * bgr[1] + 0.114 * bgr[0]
def bgr_to_ycrcb(bgr: list) -> List[int]:
    return [
        round(calculate_y(bgr)),  # Y'
        round((bgr[2] - round(calculate_y(bgr))) * 0.713 + 128),  # Cr
        round((bgr[0] - round(calculate_y(bgr))) * 0.564 + 128)  # Cb
    ]


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


def __normalize_to_zero(matrix: np.ndarray) -> np.ndarray:
    return matrix - 128


@lru_cache(maxsize=1024)
def __cos_element(x, u):
    return math.cos((2 * x + 1) * u * math.pi / 16)


@lru_cache(maxsize=1024)
def __alpha(u):
    return 1 / math.sqrt(2) if u == 0 else 1


@lru_cache(maxsize=1024)
def __g_uv(u, v, matrix: Tuple[tuple]):
    return (1 / 4) * __alpha(u) * __alpha(v) * sum(
        matrix[x][y] * __cos_element(x, u) * __cos_element(y, v)
        for x in range(len(matrix))
        for y in range(len(matrix[0])))



def __tuple_wrapper(matrix: np.ndarray) -> Tuple[tuple]:
    return tuple(map(tuple, matrix))

def __discrete_cosine_transform(matrix: np.ndarray) -> np.ndarray:
    return np.array([[__g_uv(y, x, __tuple_wrapper(matrix))
                      for x in range(len(matrix[y]))]
                     for y in range(len(matrix))])



def quantization(submatrix: np.ndarray) -> np.ndarray:
    return np.array([[round(submatrix[row, col] / quantization_matrices[submatrix.shape[0]][row, col])
                      if quantization_matrices[submatrix.shape[0]][row, col] != 0
                      else 0
                      for col in range(len(submatrix[row]))]
                     for row in range(len(submatrix))])



def dct(matrix: np.ndarray) -> np.ndarray:
    return __discrete_cosine_transform(__normalize_to_zero(matrix))
def RLE(img):
    bits=8
    encoded = []
    shape=img.shape
    count = 0
    prev = None
    fimg = img.flatten()
    th=127
    for pixel in fimg:
        if prev==None:
            prev = pixel
            count+=1
        else:
            if prev!=pixel:
                encoded.append((count, prev))
                prev=pixel
                count=1
            else:
                if count<(2**bits)-1:
                    count+=1
                else:
                    encoded.append((count, prev))
                    prev=pixel
                    count=1
    encoded.append((count, prev))
   
    return np.array(encoded)

def encode_image(src_path, dest_path, size=8) -> bool:
    print("Reading the file")
    bitmap =  cv2.imread(src_path)
    entropy=False
    if entropy:
        print("Bitmap entropy: " + str(entropy(bitmap)))

   
    bitmap = crop_bitmap(bitmap)

    print("Converting RGB to YCrCb")
    bitmap = np.apply_along_axis(bgr_to_ycrcb, 2, bitmap)
    print(len(bitmap))
    #print(bitmap)
    bp=np.array(bitmap)
    print(bp.shape)
    print("Separating bitmap to Y, Cb, Cr matrices")
    y, cb, cr = split_to_ycbcr(bitmap)

    print("Downsampling")
    cr = chromasubsample(cr)
    cb = chromasubsample(cb)

    y_shape = shape_for_contacting(y.shape, size)
    cr_shape = shape_for_contacting(cr.shape, size)
    cb_shape = shape_for_contacting(cb.shape, size)
    # print(y_shape, y.shape)
    # print(cr_shape, cr.shape)
    # print(cb_shape, cb.shape)



    shapedata=[y_shape,cr_shape,cb_shape,bp.shape]
    file = open('shapedata', 'wb')
    pickle.dump(shapedata, file)
    file.close()



    print("Splitting to {0}x{0} blocks".format(size))
    y = split_to_blocks(y, size)
    cr = split_to_blocks(cr, size)
    cb = split_to_blocks(cb, size)

    print("dct")
    y = [dct(sub_matrix) for sub_matrix in y]
    cr = [dct(sub_matrix) for sub_matrix in cr]
    cb = [dct(sub_matrix) for sub_matrix in cb]

    print("Quantization")
    y = [quantization(submatrix) for submatrix in y]
    cr = [quantization(submatrix) for submatrix in cr]
    cb = [quantization(submatrix) for submatrix in cb]
    
    if entropy:
        print("Compressed entropy: " + str(
            entropy(
                np.dstack([np.dstack(y), np.dstack(cr), np.dstack(cb)]))))
   
    data=[y,cb,cr]
    file = open('sendfile', 'wb')
    pickle.dump(data, file)

# close the file
    file.close()
    

   
