#!/usr/bin/python3

import argparse

import arff

parser = argparse.ArgumentParser()
parser.add_argument('infile', nargs='+')
parser.add_argument('outfile')
parser.add_argument(
    '-s', '--safe', help="Check ARFF structure", default=False, action='store_true')


def main():
    args = parser.parse_args()
    files = args.infile
    if len(files) < 2:
        raise ValueError("Please specify at least two input files")

    if args.safe:
        data_list = []
        for file in files:
            with open(file) as fid:
                data_list.append(arff.load(fid))

        if not all([d['attributes'] == data_list[0]['attributes'] for d in data_list]):
            raise ValueError("Some data attributes are different")

        data = data_list[0]
        for d in data_list[1:]:
            data['data'].extend(d['data'])

        with open(args.outfile, 'w') as fid:
            arff.dump(data, fid)
    else:
        with open(files[0]) as fid:
            lines = fid.readlines()
        for file in files[1:]:
            with open(file) as fid:
                for line in fid:
                    if not line.isspace() and not line.startswith('@'):
                        lines.append(line)

        with open(args.outfile, 'w') as fid:
            fid.writelines(lines)


if __name__ == "__main__":
    main()
