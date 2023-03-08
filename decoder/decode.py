import pickle
import numpy as np
import sys
import argparse
from functions import *

def main():
    parser = argparse.ArgumentParser(
        description='Jpeg decoder')
    parser.add_argument(
        'SRC',
        help='compressed file'
    )
    parser.add_argument(
        'DST',
        help=
        'Path to directry to save file and name',
        default='./')
    args = parser.parse_args()
    decode_image(args.SRC, args.DST)
    


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())