import hashlib
import pathlib


def cache_key(model: str, image_hash: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}|{image_hash}|{prompt}".encode()).hexdigest()


def image_hash(path: str) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
