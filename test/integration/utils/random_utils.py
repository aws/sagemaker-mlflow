import uuid
import random

""" Generates a random uuid that can be 64 characters or shorter.
"""


def generate_uuid(num_chars: int) -> str:
    if num_chars > 64:
        raise Exception("Must be 64 characters or under")
    return str(uuid.uuid4())[:num_chars]


""" Generates a random floating point integer
"""


def generate_random_float() -> float:
    return random.uniform(1.0, 20.0)
