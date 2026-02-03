import re
from collections import defaultdict
from multiprocessing import Pool
import time

import regex
class tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges

        self.reverse_vocab = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
        self.merge_priority = {pair: id for id, pair in enumerate(merges)}

        if special_tokens == None:
            self.special_tokens = set()
        else:
            self.special_tokens = set(special_tokens)
            for st in self.special_tokens:
                st_bytes = st.encode("utf-8")
                if st_bytes not in self.reverse_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = st_bytes
                    self.reverse_vocab[st_bytes] = new_id

    def encode(self, text: str) -> list[int]:

        if self.special_tokens:
            split_pattern = "(" + "|".join(re.escape(st) for st in self.special_tokens) + ")"
            segements = re.split(split_pattern, text)
        else:
            segements = [text]

        all_ids = []
        
        for seg in segements:
            if seg in self.special_tokens:
                all_ids.append(self.reverse_vocab[seg.encode("utf-8")])
            else:


            