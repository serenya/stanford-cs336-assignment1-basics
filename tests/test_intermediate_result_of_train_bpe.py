import json
import time
from .common import FIXTURES_PATH

from .adapters import run_train_bpe

def test_train_bpe():
    input_path = FIXTURES_PATH / "lowlowerwidestnewest.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=270,
        special_tokens=["<|endoftext|>"],
    )

    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        
        assert b"<|" not in word_bytes
