"""Resets the 'corpus' attribute of the dataset exported by the original
auDeep code. This assumes that the dataset was originally generated from
spectrograms extracted by our script, not auDeep's, because theirs
doesn't include proper label information.
"""

import argparse

import netCDF4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('corpus')
    args = parser.parse_args()

    dataset = netCDF4.Dataset(args.filename, 'a')
    dataset.setncattr_string('corpus', args.corpus)
    dataset.close()
    print("Changed corpus to {} in {}".format(args.corpus, args.filename))


if __name__ == "__main__":
    main()
