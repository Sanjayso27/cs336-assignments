from typing import Iterable
from collections import Counter
import regex as re

def find_pretokens(text: str) -> Counter:
    """
    Find the pretokens in the text.
    Args:
        text: The input text string.
    Returns:
        A Counter object mapping pre-tokens to their frequencies.
    """
    GPT2_PRETOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    pretokens = re.findall(GPT2_PRETOKENIZER_PATTERN, text)
    return Counter(pretokens)

def read_input_file(input_path: str, special_tokens: Iterable[str]) -> Counter:
    """
    Read the input text file and return a frequency table of pre-tokenized byte tuples.
    Args:
        input_path: Path to the input text file.
        special_tokens: List of special tokens to consider during tokenization.
    Returns:
        A Counter object mapping byte tuples to their frequencies.
    """
    # Read the input text file
    with open(input_path, "r") as file:
        text = file.read()

    # Remove special tokens from the text
    for token in special_tokens:
        text = text.replace(token, "")
    
    pretokens = find_pretokens(text)
    gen_tuple_of_bytes = lambda pretoken: tuple([bytes([b]) for b in pretoken.encode("utf-8")])
    pretoken_freq = {}
    for pretoken, freq in pretokens.items():
        pretoken_freq[gen_tuple_of_bytes(pretoken)] = freq
    
    return pretoken_freq

def update_byte_tuple(byte_tuple: Iterable[bytes], merge_loc: int) -> tuple[Iterable[bytes], Iterable[bytes], Iterable[bytes]]:
    """
    Merge the byte tuple at the merge location.
    Args:
        byte_tuple: The original byte tuple.
        merge_loc: The location to merge the bytes.
    Returns:
        A tuple containing the new byte tuple, prefix, and suffix.
    """
    assert len(byte_tuple) > 1, "Cannot merge a byte tuple with length less than 2."
    prefix = byte_tuple[:merge_loc]
    tomerge = byte_tuple[merge_loc:merge_loc+2]
    suffix = byte_tuple[merge_loc+2:]
    new_byte_tuple = prefix + (b"".join(tomerge),) + suffix
    return new_byte_tuple, prefix, suffix

def train_bpe(input_path: str, vocab_size: int, special_tokens: Iterable[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte pair encoding tokenizer on the input text file.

    Args:
        input_path: Path to the input text file.
        vocab_size: Size of the vocabulary.
        special_tokens: List of special tokens to add to the vocabulary.

    Returns:
        Tuple of the learned vocab and merges.
    """
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256+i] = token.encode("utf-8")

    pretoken_freq = read_input_file(input_path, special_tokens)

    pair_freq = Counter()
    for pretoken, freq in pretoken_freq.items():
        for i in range(len(pretoken)-1):
            pair = pretoken[i:i+2]
            pair_freq[pair] += freq
    
    #BPE algorithm
    merges = []
    while len(vocab) < vocab_size:
        #find the most frequent pair
        most_freq_pair = max(pair_freq, key=lambda p: (pair_freq[p], p))

        # Add the pair to the merges list
        merges.append(most_freq_pair)

        # Update the vocab
        new_id = max(vocab.keys()) + 1
        vocab[new_id] = b"".join(most_freq_pair)

        # Update the pre-token frequency table and pair frequency table
        new_pretoken_freq = {}
        for pretoken, freq in pretoken_freq.items():
            i=0
            while(i<len(pretoken)):
                pair = pretoken[i:i+2]
                if pair == most_freq_pair:
                    pretoken, prefix, suffix = update_byte_tuple(pretoken, i)

                    # Update pair frequencies for the prefix
                    if len(prefix)>0:
                        add_pair = (prefix[-1], vocab[new_id])
                        pair_freq[add_pair] += freq
                        del_pair = (prefix[-1], pair[0])
                        pair_freq[del_pair] -= freq
                    # Update pair frequencies for the suffix
                    if len(suffix)>0:
                        add_pair = (vocab[new_id], suffix[0])
                        pair_freq[add_pair] += freq
                        del_pair = (pair[1], suffix[0])
                        pair_freq[del_pair] -= freq
                    pair_freq[most_freq_pair] -= freq
                i += 1
            # Update the new pretoken frequency table
            new_pretoken_freq[pretoken] = freq
        pretoken_freq = new_pretoken_freq
    return vocab, merges
