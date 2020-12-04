import argparse
import json
from pathlib import Path

import jsonschema
import requests

SCHEMA_URI = 'https://agkphysics.github.io/schemas/emotional-speech-data/'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--version', type=str, default='latest')
    args = parser.parse_args()

    schema_uri = SCHEMA_URI + args.version
    schema = json.loads(requests.get(schema_uri).content)

    with open(args.input) as fid:
        data = json.load(fid)
    jsonschema.validate(data, schema)
    print("Metadata successfully validated.")


if __name__ == "__main__":
    main()
