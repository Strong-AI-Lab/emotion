import argparse
import struct
from io import RawIOBase

import arff

__all__ = ['encode', 'decode']

tp_int = {
    'numeric': 1,
    'string': 2,
    'date': 3,
    'relational': 4
}

int_tp = {
    1: 'numeric',
    2: 'string',
    3: 'date',
    4: 'relational'
}

MAX_RELATION_LEN = 256
RELATION_FMT = '<{}s'.format(MAX_RELATION_LEN)
MAX_ATTR_LEN = 256
ATTR_FMT = '<{}sB'.format(MAX_ATTR_LEN)
MAX_NOM_LEN = 64
NOM_FMT = '{}s'.format(MAX_NOM_LEN)


def remove_null(b: bytes) -> str:
    b = b.decode()
    return b[:b.find('\x00')]


def encode(fid: RawIOBase, data: dict):
    fid.write(struct.pack(RELATION_FMT, data['relation'].encode()))

    fid.write(struct.pack('<I', len(data['attributes'])))
    packer = struct.Struct(ATTR_FMT)
    for name, tp in data['attributes']:
        fid.write(packer.pack(
            name.encode(), 0 if isinstance(tp, list) else tp_int[tp.lower()]))
        if isinstance(tp, list):
            fmt_str = '<I' + NOM_FMT * len(tp)
            fid.write(struct.pack(fmt_str, len(tp), *[x.encode() for x in tp]))

    fmt_str = '<'
    for _, tp in data['attributes']:
        if not isinstance(tp, list) and tp.lower() == 'numeric':
            fmt_str += 'f'
        else:
            fmt_str += NOM_FMT
    packer = struct.Struct(fmt_str)
    for inst in data['data']:
        inst = [x.encode() if isinstance(x, str) else x for x in inst]
        fid.write(packer.pack(*inst))


def decode(fid: RawIOBase):
    data = {}
    relation = struct.unpack(RELATION_FMT, fid.read(MAX_RELATION_LEN))[0]
    data['relation'] = remove_null(relation)

    data['attributes'] = []
    tp_array = []
    num_attrs = struct.unpack('<I', fid.read(4))[0]
    packer = struct.Struct(ATTR_FMT)
    for i in range(num_attrs):
        name, tp = packer.unpack(fid.read(packer.size))
        name = remove_null(name)

        tp_array.append(tp)
        if tp == 0:
            num_tps = struct.unpack('<I', fid.read(4))[0]
            fmt_str = NOM_FMT * num_tps
            tps = [remove_null(x) for x in struct.unpack(
                fmt_str, fid.read(struct.calcsize(fmt_str)))]
            data['attributes'].append((name, tps))
        else:
            data['attributes'].append((name, int_tp[tp].upper()))

    fmt_str = '<'
    for tp in tp_array:
        if tp == 1:
            fmt_str += 'f'
        else:
            fmt_str += NOM_FMT
    packer = struct.Struct(fmt_str)

    data['data'] = []
    while True:
        buf = fid.read(packer.size)
        if len(buf) == 0:
            break
        inst = packer.unpack(buf)
        inst = [remove_null(x) if isinstance(x, bytes) else x for x in inst]
        data['data'].append(inst)
    return data


parser = argparse.ArgumentParser()
parser.add_argument('infile')
parser.add_argument('outfile')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-e', '--encode', help="Encode ARFF to binary", action='store_true')
group.add_argument('-d', '--decode', help="Decode ARFF from binary", action='store_true')


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
