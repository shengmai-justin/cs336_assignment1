import re
from collections import defaultdict
from multiprocessing import Pool
import time

import regex

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def get_pair_counts(pretokens: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    """Count all adjacent pairs across all pre-tokens, weighted by frequency."""
    pair_count: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for pre_token, freq in pretokens.items():
        for i in range(len(pre_token) - 1):
            pair = (pre_token[i], pre_token[i + 1])
            pair_count[pair] += freq
        
    return pair_count

def merge_pair_in_tuple(token_tuple: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:

    results = []
    i = 0
    while i < len(token_tuple) - 1:
        current_pair = (token_tuple[i], token_tuple[i + 1])
        if current_pair == pair:
            results.append(token_tuple[i] + token_tuple[i + 1])
            i += 2
        else:
            results.append(token_tuple[i])
            i += 1
    
    if i < len(token_tuple):
        results.append(token_tuple[i])

    return tuple(results)

def apply_merge(pretokens: dict[tuple[bytes, ...], int], pair: tuple[bytes, bytes]) -> dict[tuple[bytes, ...], int]:

    new_pretokens: dict[tuple[bytes, ...], int] = {}

    for token_tuple, freq in pretokens.items():

        if len(token_tuple) < 2:
            new_pretokens[token_tuple] = new_pretokens.get(token_tuple, 0) + freq
            continue

        new_tuple = merge_pair_in_tuple(token_tuple, pair)
        new_pretokens[new_tuple] = new_pretokens.get(new_tuple, 0) + freq

    return new_pretokens

def apply_merge_and_update_counts(
    pretokens: dict[tuple[bytes, ...], int],
    pair_to_merge: tuple[bytes, bytes],
    pair_counts: dict[tuple[bytes, bytes], int]
) -> tuple[dict[tuple[bytes, ...], int], dict[tuple[bytes, bytes], int]]:
    
    new_pretokens: dict[tuple[bytes, ...], int] = {}

    for token_tuple, freq in pretokens.items():
        if len(token_tuple) < 2:
            new_pretokens[token_tuple] = new_pretokens.get(token_tuple, 0) + freq
            continue

        new_token_tuple = merge_pair_in_tuple(token_tuple, pair_to_merge)
        
        if new_token_tuple == token_tuple:
            # No merge happened, keep as-is
            new_pretokens[token_tuple] = freq
            continue
            
        for i in range(len(token_tuple) - 1):
            old_pair = (token_tuple[i], token_tuple[i+1])
            pair_counts[old_pair] -= freq
            if pair_counts[old_pair] <= 0:  # ← Add this check
                del pair_counts[old_pair]
        
        for i in range(len(new_token_tuple) - 1):
            new_pair = (new_token_tuple[i], new_token_tuple[i+1])
            pair_counts[new_pair] = pair_counts.get(new_pair, 0) + freq
        
        new_pretokens[new_token_tuple] = new_pretokens.get(new_token_tuple, 0) + freq

    
    return new_pretokens, pair_counts

        





def pretokenize(input_path: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
        split_pattern = "|".join(re.escape(st) for st in special_tokens)
        segements = re.split(split_pattern, text)

        counts: dict[tuple[bytes, ...], int] = {}

        for seg in segements:
            for match in regex.finditer(PAT, seg):
                word = match.group()

                byte_tuple = tuple(bytes([b]) for b in word.encode("utf-8"))
                counts[byte_tuple] = counts.get(byte_tuple, 0) + 1
        
        return counts

def pretokenize_chunk(
    file_path: str,
    start: int,
    end: int,
    special_tokens: list[str],
) -> dict[tuple[bytes, ...], int]:
        with open(file_path, 'rb') as f:
            f.seek(start)
            chunk_bytes = f.read(end - start) #read the (end-start) # of bytes
            text = chunk_bytes.decode('utf-8', errors="ignore") #decode because we convert bytes to string
            split_pattern = "|".join(re.escape(st) for st in special_tokens)
            segements = re.split(split_pattern, text)

            counts: dict[tuple[bytes, ...], int] = {}

            for seg in segements:
                for match in regex.finditer(PAT, seg):
                    word = match.group()
                    byte_tuple = tuple(bytes([b]) for b in word.encode("utf-8")) # we encode to convert string to bytes. 
                    counts[byte_tuple] = counts.get(byte_tuple, 0) + 1
            
            return counts

def pretokenize_parallel(
    input_path: str,
    special_tokens: list[str],
    num_workers: int = None,
) -> dict[tuple[bytes, ...], int]:
    import multiprocessing
    if num_workers == None:
        num_workers = multiprocessing.cpu_count()

    with open(input_path, 'rb') as f:
        split_tokens = special_tokens[0]
        split_tokens_bytes = split_tokens.encode("utf-8")
        boundaries = find_chunk_boundaries(
            file=f,
            split_special_token=split_tokens_bytes,
            desired_num_chunks=num_workers
        )
        chunks = list(zip(boundaries[:-1], boundaries[1:]))
        worker_args = [(input_path, start, end, special_tokens) for start, end in chunks]
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        chunk_results = pool.starmap(pretokenize_chunk, worker_args)

    combined_counts: dict[tuple[bytes, ...], int] = {}
    for chunk in chunk_results:
        for b, count in chunk.items():
            combined_counts[b] = combined_counts.get(b, 0) + count

    return combined_counts

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    print("=== BPE Tokenizer Training ===")
    t_start = time.time()

    vocab: dict[int, bytes] = {}
    for i in range(256):
        vocab[i] = bytes([i])
    
    for i, st in enumerate(special_tokens):
        vocab[256 + i] = st.encode("utf-8")

    num_merges_target = vocab_size - 256 - len(special_tokens)
    print(f"  Vocab init: 256 bytes + {len(special_tokens)} special tokens. Need {num_merges_target} merges.")

    merges: list[tuple[bytes, bytes]] = []
    pretokens = pretokenize_parallel(input_path=input_path, special_tokens=special_tokens)

    pair_counts = get_pair_counts(pretokens)

    for merge_idx in range(num_merges_target):
        #pair_counts = get_pair_counts(pretokens)
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merges.append(best_pair)

        new_id = 256 + len(special_tokens) + merge_idx
        vocab[new_id] = best_pair[0] + best_pair[1]
        pretokens, pair_counts = apply_merge_and_update_counts(pretokens=pretokens, pair_to_merge=best_pair, pair_counts=pair_counts)

    t_end = time.time()
    print(f"  Training completed in {t_end - t_start:.2f}s")
    
    return (vocab, merges)


