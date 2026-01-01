from typing import Iterable
import regex as re

from utils.io import get_tokenizer_from_vocab_merges_path, GPT2_PRETOKENIZER_PATTERN


def _fix_vocab(vocab_i_to_b, vocab_b_to_i):
        """ Make sure all bytes are in the vocab """
        for i in range(256):
            byte = bytes([i])
            if byte not in vocab_b_to_i:
                vocab_b_to_i[byte] = len(vocab_b_to_i)
                vocab_i_to_b[len(vocab_i_to_b)] = byte
        return dict(int_to_byte=vocab_i_to_b, byte_to_int=vocab_b_to_i)

def _get_pairs(token_ids: list[int]) -> set[tuple[int, int]]:
    """ Return a set of unique adjacent token ID pairs from the input list. """
    pairs = set()
    for i in range(len(token_ids) - 1):
        pairs.add((token_ids[i], token_ids[i + 1]))
    return pairs

def _update(ids: list[int], pair: tuple[int,int], new_id:int)-> list[int]:
    """Update the ids by merging the pairs """
    new_ids = []
    i=0
    while i<len(ids):
        cur_pair = tuple(ids[i:i+2])
        if cur_pair == pair:
            new_ids.append(new_id)
            i+=1
        else :
            new_ids.append(ids[i])
        i+=1
    return new_ids

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = {}
        self.vocab['int_to_byte'] = vocab
        self.vocab['byte_to_int'] = {v: k for k, v in vocab.items()}
        self.vocab = _fix_vocab(self.vocab['int_to_byte'], self.vocab['byte_to_int'])
        
        # reorganzie merges into pair -> new token id dict
        self.merges = {}
        for a, b in merges:
            id_pair = (self.vocab['byte_to_int'][a], self.vocab['byte_to_int'][b])
            self.merges[id_pair] = self.vocab['byte_to_int'][a+b]

        # add special tokens as string to id mapping
        self.special_tokens = {}
        if special_tokens:
            special_tokens = sorted(special_tokens, key=len, reverse=True)
            for token in special_tokens:
                token_byte = token.encode("utf-8")
                if token_byte not in self.vocab['byte_to_int']:
                    self.vocab['byte_to_int'][token_byte] = len(self.vocab['byte_to_int'])
                    self.vocab['int_to_byte'][len(self.vocab['int_to_byte'])] = token_byte
                    self.special_tokens[token] = len(self.vocab['int_to_byte'])
                else:
                    self.special_tokens[token] = self.vocab['byte_to_int'][token_byte]

    def _encode_chunk(self, text: str) -> list[int]:
        """
        Encode a chunk of text (without special tokens) into a list of token IDs using BPE merges.
        """
        if text in self.special_tokens:
            return [self.special_tokens[text]]
        else :
            text_chunks = re.findall(GPT2_PRETOKENIZER_PATTERN, text)
            token_ids = []
            for chunk in text_chunks:
                text_bytes = chunk.encode("utf-8")
                ids = [self.vocab['byte_to_int'][bytes([b])] for b in text_bytes]
                while len(ids) > 1:
                    pairs = _get_pairs(ids)
                    best_pair = min(pairs, key=lambda pair:self.merges.get(pair,float('inf')))
                    if best_pair not in self.merges:
                        break
                    new_id = self.merges[best_pair]
                    ids = _update(ids,best_pair, new_id)
                token_ids.extend(ids)
            return token_ids
                    

    def encode(self, text: str) -> list[int]:
        """
        Encode the input text into a list of token IDs using the BPE merges and vocabulary.

        Args:
            text: The input text string to encode.
        Returns:
            A list of token IDs representing the encoded text.
        """
        if self.special_tokens:
            special_pattern = "(" + "|".join(re.escape(token) for token in self.special_tokens) + ")"
            special_split_chunk = re.split(special_pattern, text)
        else:
            special_split_chunk = [text]
        ids = []
        for chunk in special_split_chunk:
            ids += self._encode_chunk(chunk)
        return ids

    def encode_iterable(self, texts: Iterable[str]) -> Iterable[int]:
        """
        Encode an iterable of text strings into a list of lists of token IDs.

        Args:
            iterable: An iterable of text strings to encode.
        Returns:
            A list where each element is a list of token IDs for the corresponding input string.
        """
        for text in texts:
            ids = self.encode(text)
            for id in ids:
                yield id

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode a list of token IDs back into the original text string using the vocabulary.

        Args:
            token_ids: A list of token IDs to decode.
        Returns:
            The decoded text string.
        """
        text_bytes = b''.join([self.vocab['int_to_byte'][i] for i in token_ids])
        return text_bytes.decode("utf-8", errors="replace")

    def from_files(cls, vocab_path: str, merges_path: str, special_tokens: list[str] | None = None):
        """
        Create a Tokenizer instance by loading the vocabulary and merges from files.

        Args:
            vocab_path: Path to the vocabulary file.
            merges_path: Path to the merges file.
            special_tokens: Optional list of special tokens to include in the tokenizer.
        Returns:
            An instance of the Tokenizer class.
        """
        vocab, merges = get_tokenizer_from_vocab_merges_path(vocab_path, merges_path)
        return cls(vocab, merges, special_tokens)