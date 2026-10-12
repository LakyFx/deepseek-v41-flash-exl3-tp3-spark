"""Download the hash-pinned upstream client as data, never execute its CLI."""
from pathlib import Path
import hashlib
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / '.cache/owen/bin-spark-dd35633.py'
URL = 'https://raw.githubusercontent.com/christopherowen/spark-ds41f/dd35633edeadd8864ec177890025b6e7c5df47ce/bin/spark'
SHA256 = 'b130f172ec4e039532e9fb0b7d4d6256ddc5f16918c88d0689841146631ba46a'


def main():
    raw = DEST.read_bytes() if DEST.exists() else urllib.request.urlopen(URL, timeout=60).read()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError('upstream source differs from the measured revision')
    if not DEST.exists():
        DEST.parent.mkdir(parents=True, exist_ok=True)
        with DEST.open('xb') as stream:
            stream.write(raw)
    print(str(DEST))


if __name__ == '__main__':
    main()
