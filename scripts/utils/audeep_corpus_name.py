"""Resets the 'corpus' attribute of the dataset exported by the original
auDeep code. This assumes that the dataset was originally generated from
spectrograms extracted by our script, not auDeep's, because theirs
doesn't include proper label information.
"""

import sys

import netCDF4


def main():
    dataset = netCDF4.Dataset(sys.argv[1], 'a')
    dataset.setncattr_string('corpus', sys.argv[2])
    if 'slices' not in dataset.variables:
        slices = dataset.createVariable('slices', int, ('instance',))
        slices[:] = [1] * dataset.dimensions['instance'].size
    dataset.close()
    print(f"Changed corpus to {sys.argv[2]} in {sys.argv[1]}")


if __name__ == "__main__":
    main()
