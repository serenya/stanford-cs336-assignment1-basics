import json
import time

from .adapters import run_train_bpe
from .common import FIXTURES_PATH


def test_train_bpe_openwebtext(snapshot):
    """
    Ensure that the special tokens are added to the vocabulary and not
    merged with other tokens.
    """
    input_path = FIXTURES_PATH / "data/owt_train.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
    )
