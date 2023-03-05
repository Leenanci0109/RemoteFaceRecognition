import numpy as np
import math
from functools import lru_cache
from typing import Tuple, List
import cv2
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
def __un_normalize(matrix: np.ndarray):
    return matrix + 128

def __tuple_wrapper(matrix: np.ndarray) -> Tuple[tuple]:
    return tuple(map(tuple, matrix))

@lru_cache(maxsize=1024)
def __cos_element(x, u):
    return math.cos((2 * x + 1) * u * math.pi / 16)


@lru_cache(maxsize=1024)
def __alpha(u):
    return 1 / math.sqrt(2) if u == 0 else 1
@lru_cache(maxsize=512)
def __f_xy(x, y, matrix: Tuple[Tuple[float]]):
    return round(
        0.25 *
        sum(
            __alpha(u) * __alpha(v) * matrix[v][u] *
            __cos_element(x, u) * __cos_element(y, v)
            for u in range(len(matrix[0]))
            for v in range(len(matrix)))
    )

def __invert_discrete_cosine_transform(matrix: np.ndarray):
    return np.array([[__f_xy(x, y, __tuple_wrapper(matrix))
                      for x in range(len(matrix[y]))]
                     for y in range(len(matrix))])

def inverse_dct(matrix):
    return __un_normalize(__invert_discrete_cosine_transform(matrix))

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

def RLE_decode(encoded, shape):
    decoded=[]
    for rl in encoded:
        r,p = rl[0], rl[1]
        decoded.extend([p]*r)
    dimg = np.array(decoded).reshape(shape)
    return dimg


def ycrcb_to_bgr(matrix3d: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(ycrcb_pixel_to_bgr, 2, matrix3d)

def save_img(matrix: np.ndarray, mode: str = 'RGB', dest: str = 'tmp.jpeg'):
    if mode == 'YCrCb':
        matrix = ycrcb_to_bgr(matrix)
    cv2.imwrite(dest, matrix)


