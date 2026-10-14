import hashlib
import re
import numpy as np

STOP = set('a an the is are was were i you my your me to for in on at of and or it that this with have has had'.split())


def key_vector(text, dimensions):
    vector = np.zeros(dimensions, dtype=np.float32)
    for word in re.findall(r'[a-z]+', text.lower()):
        if word in STOP:
            continue
        value = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), 'little')
        vector[value % dimensions] += 1 if (value // dimensions) % 2 else -1
    norm = np.linalg.norm(vector)
    return (vector / norm if norm else vector).astype('<f2')


def inferred_as_of(query):
    if 'as_of' in query:
        return query['as_of']
    # Only explicit historical phrases; the benchmark query date is not a past-version request.
    match = re.search(r'\b(?:in|during|as of|back in)\s+((?:19|20)\d{2})(?:\b)', query['question'], re.I)
    return match.group(1) + '-12-31T23:59:59' if match else None
