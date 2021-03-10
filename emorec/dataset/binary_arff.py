import struct
from io import RawIOBase

__all__ = ['encode', 'decode']

_tp_int = {
    'numeric': 1,
    'string': 2,
    'date': 3,
    'relational': 4
}

_int_tp = {
    1: 'numeric',
    2: 'string',
    3: 'date',
    4: 'relational'
}

MAX_RELATION_LEN = 256
RELATION_FMT = f'<{MAX_RELATION_LEN}s'
MAX_ATTR_LEN = 256
ATTR_FMT = f'<{MAX_ATTR_LEN}sB'
MAX_NOM_LEN = 64
NOM_FMT = f'{MAX_NOM_LEN}s'


def _remove_null(b: bytes) -> str:
    s = b.decode()
    return s[:s.find('\x00')]


def encode(fid: RawIOBase, data: dict):
    """Encodes a text ARFF file to a binary file with essentially the same
    similar structure.

    Parameters:
    -----------
    fid: a writeable file object
        The file handle to write to

    data: dict
        The ARFF data dictionary. Must have 'relation', 'attributes' and 'data'
        keys.
    """
    fid.write(struct.pack(RELATION_FMT, data['relation'].encode()))

    fid.write(struct.pack('<I', len(data['attributes'])))
    packer = struct.Struct(ATTR_FMT)
    for name, tp in data['attributes']:
        fid.write(packer.pack(
            name.encode(), 0 if isinstance(tp, list) else _tp_int[tp.lower()]))
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
    """Decodes a binary ARFF file that was created with encode().

    Parameters:
    -----------
    fid: file object
        The file handle to read binary data from.

    Returns:
    --------
    data: dict
        A dictionary representing the ARFF file.
    """
    data = {}
    relation = struct.unpack(RELATION_FMT, fid.read(MAX_RELATION_LEN))[0]
    data['relation'] = _remove_null(relation)

    data['attributes'] = []
    tp_array = []
    num_attrs = struct.unpack('<I', fid.read(4))[0]
    packer = struct.Struct(ATTR_FMT)
    for i in range(num_attrs):
        name, tp = packer.unpack(fid.read(packer.size))
        name = _remove_null(name)

        tp_array.append(tp)
        if tp == 0:
            num_tps = struct.unpack('<I', fid.read(4))[0]
            fmt_str = NOM_FMT * num_tps
            tps = [_remove_null(x) for x in struct.unpack(
                fmt_str, fid.read(struct.calcsize(fmt_str)))]
            data['attributes'].append((name, tps))
        else:
            data['attributes'].append((name, _int_tp[tp].upper()))

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
        inst = [_remove_null(x) if isinstance(x, bytes) else x for x in inst]
        data['data'].append(inst)
    return data
