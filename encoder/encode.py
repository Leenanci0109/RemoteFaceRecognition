from functions import *
import sys
import argparse
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
    parser.add_argument(
        '-s',
        '--size',
        default=8,
        type=int,
        choices=[8, 16, 32],
        help='Sub-matrix size*size')
    args = parser.parse_args()
    encode_image(args.SRC, args.DST, args.size)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())