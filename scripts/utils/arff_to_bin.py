import argparse

import arff

from emotion_recognition.binary_arff import decode, encode

parser = argparse.ArgumentParser()
parser.add_argument('infile')
parser.add_argument('outfile')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-e', '--encode', action='store_true',
                   help="Encode ARFF to binary")
group.add_argument('-d', '--decode', action='store_true',
                   help="Decode ARFF from binary")


def main():
    args = parser.parse_args()

    print("Reading")
    with open(args.infile, 'r' if args.encode else 'br') as fid:
        if args.encode:
            data = arff.load(fid)
        else:
            data = decode(fid)

    print("Writing")
    with open(args.outfile, 'bw' if args.encode else 'w') as fid:
        if args.encode:
            encode(fid, data)
        else:
            arff.dump(data, fid)


if __name__ == "__main__":
    main()
