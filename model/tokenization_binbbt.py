import os
import json
import base64
import re
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import transformers
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}


def replace_numbers(text):
    # Helper function to replace found numbers in text with their binary representation
    def replace(match):
        # Extract the matched number from the text
        number = match.group()
        if '.' in number:  # If the number is a float
            try:
                # Pack the float number into binary (float32)
                packed = struct.pack('f', float(number))
            except (OverflowError, ValueError, struct.error):
                # Handle overflow/underflow by using the max or min float32 values
                if float(number) > 0:
                    packed = struct.pack('f', np.finfo(np.float32).max)
                else:
                    packed = struct.pack('f', np.finfo(np.float32).min)
            return packed
        else:  # If the number is an integer
            try:
                # Pack the integer into binary (int32)
                packed = struct.pack('i', int(number))
            except (OverflowError, ValueError, struct.error):
                # Handle overflow/underflow by using the max or min int32 values
                if int(number) > 0:
                    packed = struct.pack('i', np.iinfo(np.int32).max)
                else:
                    packed = struct.pack('i', np.iinfo(np.int32).min)
            return packed

    # Initialize a bytearray to store the processed text
    parts = bytearray()
    last_end = 0
    # Iterate through all found numbers in the text using regex
    for match in re.finditer(r'\d+\.\d+|\d+', text):
        # Append the non-number parts of the text
        if match.start() > last_end:
            parts.append(text[last_end:match.start()])
        # Replace the number with its binary representation
        parts.append(replace(match))
        last_end = match.end()
    # Append the remaining text after the last match
    if last_end < len(text):
        parts.append(text[last_end:])
    
    return parts


def utf8_decode(tokens) -> str:
    # Decode a list of UTF-8 encoded bytes back into a string
    current_str = ""
    i = 0
    while i < len(tokens):
        byte = tokens[i]
        if isinstance(byte, str):
            # Directly append string characters
            current_str += byte
            i += 1
        elif isinstance(byte, int):
            try:
                # Decode ASCII byte (0-127) directly
                if byte <= 0x7F:
                    current_str += bytes([byte]).decode("utf-8")
                    i += 1
                # Decode two-byte UTF-8 sequence
                elif 0xC0 <= byte <= 0xDF:
                    if i + 1 < len(tokens) and 0x80 <= tokens[i + 1] <= 0xBF:
                        current_str += bytes(tokens[i:i+2]).decode("utf-8")
                        i += 2
                    else:
                        # Handle invalid sequences by showing the byte in brackets
                        current_str += f"[{byte}]"
                        i += 1
                # Decode three-byte UTF-8 sequence
                elif 0xE0 <= byte <= 0xEF:
                    if (i + 2 < len(tokens) and
                        0x80 <= tokens[i + 1] <= 0xBF and
                        0x80 <= tokens[i + 2] <= 0xBF):
                        current_str += bytes(tokens[i:i+3]).decode("utf-8")
                        i += 3
                    else:
                        # Handle invalid sequences by showing the byte in brackets
                        current_str += f"[{byte}]"
                        i += 1
                # Decode four-byte UTF-8 sequence
                elif 0xF0 <= byte <= 0xF7:
                    if (i + 3 < len(tokens) and
                        0x80 <= tokens[i + 1] <= 0xBF and
                        0x80 <= tokens[i + 2] <= 0xBF and
                        0x80 <= tokens[i + 3] <= 0xBF):
                        current_str += bytes(tokens[i:i+4]).decode("utf-8")
                        i += 4
                    else:
                        # Handle invalid sequences by showing the byte in brackets
                        current_str += f"[{byte}]"
                        i += 1
                else:
                    # Handle unknown byte sequences
                    current_str += f"[{byte}]"
                    i += 1
            except UnicodeDecodeError:
                # Handle Unicode decode errors by showing the byte in brackets
                current_str += f"[{byte}]"
                i += 1
        else:
            # Raise an exception for unsupported token types
            raise "utf8_decode: tokens not List[int] nor List[str]"

    return current_str


class BinBBTTokenizer(PreTrainedTokenizer):

    # Defines the vocabulary files and model input names
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        clean_up_tokenization_spaces=False,
        spaces_between_special_tokens=False,
        add_prefix_space=True,
        replace_numbers=False,
        padding_side="left",
        **kwargs,
    ):
        # Initialize special tokens and handle cases where they are provided as strings
        bos_token = (
            AddedToken(bos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(bos_token, str)
            else bos_token
        )
        eos_token = (
            AddedToken(eos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(eos_token, str)
            else eos_token
        )
        unk_token = (
            AddedToken(unk_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(unk_token, str)
            else unk_token
        )
        pad_token = (
            AddedToken(pad_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(pad_token, str)
            else pad_token
        )

        # Set tokenizer options, including number replacement and padding side
        self.add_prefix_space = add_prefix_space
        self.replace_numbers = replace_numbers
        self.padding_side = padding_side

        # Load the vocabulary from file or create a default encoding map
        if vocab_file:
            with open(vocab_file, "r", encoding="utf-8") as f:
                readable_encoder = json.load(f)
            self.encoder = {ord(k): v for k, v in readable_encoder.items()}
        else:
            self.encoder = {i:i for i in range(256)}  # Default: byte-to-byte encoding
        self.decoder = {v: k for k, v in self.encoder.items()}

        # Call the parent class constructor with necessary parameters
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            spaces_between_special_tokens=spaces_between_special_tokens,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

    # Returns the size of the vocabulary
    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    # Returns the vocabulary dictionary
    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    # Tokenizes the input text
    def _tokenize(self, text, **kwargs):
        if self.replace_numbers:
            # Replace numbers in the text with their binary representation
            replaced_parts = replace_numbers(text)
            # Convert the replaced parts to bytes
            text_bytes = bytearray()
            for part in replaced_parts:
                if isinstance(part, str):
                    text_bytes.extend(part.encode('utf-8'))  # Encode string parts
                else:
                    text_bytes.extend(part)  # Add binary number parts directly
        else:
            # Simply encode the text to UTF-8 bytes
            text_bytes = text.encode('utf-8')

        return list(text_bytes)

    # Convert a token to its corresponding ID in the vocabulary
    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # Convert an ID back to its corresponding token
    def _convert_id_to_token(self, index):
        return self.decoder.get(index)

    # Convert a list of tokens back into a string
    def convert_tokens_to_string(self, tokens):
        # Handle cases where a prefix space is added
        if tokens[0] == b' '[0] and self.add_prefix_space:
            tokens = tokens[1:]

        current_sub_tokens = []
        out_string = ""
        prev_is_special = False

        # Iterate through the tokens and decode them
        for i, token in enumerate(tokens):
            if token in self.all_special_tokens:
                # Append special tokens directly to the output string
                out_string += utf8_decode(current_sub_tokens) + token
                current_sub_tokens = []
                prev_is_special = True
            else:
                current_sub_tokens.append(token)
                prev_is_special = False

        out_string += utf8_decode(current_sub_tokens)
        return out_string

    # Prepare the text for tokenization by adding a prefix space if necessary
    def prepare_for_tokenization(
        self, text: str, is_split_into_words: bool = False, **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        if is_split_into_words:
            return (text, kwargs)
        if self.add_prefix_space:
            text = " " + text.strip()  # Add space before text
        return (text, kwargs)

    # Save the tokenizer vocabulary to disk
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # Save the vocabulary in a human-readable format
        readable_encoder = {chr(k): v for k, v in self.encoder.items()}
        with open(out_vocab_file, "w", encoding="utf-8") as f:
            json.dump(readable_encoder, f, indent=2, sort_keys=True, ensure_ascii=False)

        return (out_vocab_file,)

    # Add special tokens (bos and eos) around the sequence(s)
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_id = [self.bos_token_id]
        eos_token_id = [self.eos_token_id]

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    # Create token type IDs for sequences (0 for first sequence, 1 for second)
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        bos_token_id = [self.bos_token_id]
        eos_token_id = [self.eos_token_id]

        output = [0] * len(bos_token_id + token_ids_0 + eos_token_id)

        if token_ids_1 is not None:
            output += [1] * len(bos_token_id + token_ids_1 + eos_token_id)

        return output


if __name__=="__main__":
    text = 'i love u 100 u'
    text2 = 'i love u'

    # tokenizer = transformers.AutoTokenizer.from_pretrained(".", trust_remote_code=True)
    tokenizer = BinBBTTokenizer()
    tokens1 = tokenizer.tokenize(text)
    print(tokens1)

    tokenized = tokenizer.batch_encode_plus(
        [text, text2],
        add_special_tokens=True,
    )
    print(tokenized)

    padded = tokenizer.pad(tokenized)
    print(padded)

    decoded = tokenizer.batch_decode(padded["input_ids"],
                                     skip_special_tokens=False)
    print(decoded)

    tokenizer.save_pretrained(".")