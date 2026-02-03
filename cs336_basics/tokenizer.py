class tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges

        self.reverse_vocab = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
        self.merge_priority = {pair: id for id, pair in enumerate(merges)}

        self.special_tokens = set(special_tokens)
        for token in special_tokens:
            