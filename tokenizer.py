import os
import sentencepiece as spm
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from tokenizers import Tokenizer
from pathlib import Path
from typing import Dict
from jinja2 import Template
import json

class TokenizerInterface:
    def __init__(self, model_path):
        self.model_path = model_path

    def encode(self, text):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def decode(self, tokens):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def bos_id(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def eos_id(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

class SentencePieceWrapper(TokenizerInterface):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.processor = spm.SentencePieceProcessor(str(model_path))

    def encode(self, text):
        return self.processor.EncodeAsIds(text)

    def decode(self, tokens):
        return self.processor.DecodeIds(tokens)

    def bos_id(self):
        return self.processor.bos_id()

    def eos_id(self):
        return self.processor.eos_id()

class HFJsonTokenizerWrapper:
    def __init__(self, model_path):
        model_path = Path(model_path)
        assert model_path.exists(), f"Tokenizer file not found: {model_path}"
        self.tokenizer = Tokenizer.from_file(str(model_path))

        self._bos_id = 151646
        self._eos_id = 151643

        config_path = model_path.parent / "tokenizer_config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                self.chat_template = config.get("chat_template", None)
        else:
            self.chat_template = None
    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def bos_id(self):
        return self._bos_id

    def eos_id(self):
        return self._eos_id

    def render_chat(self, messages, add_generation_prompt=True):
        if self.chat_template is None:
            raise ValueError("No chat_template found in tokenizer_config.json")

        template = Template(self.chat_template, trim_blocks=True, lstrip_blocks=True)
        return template.render(messages=messages, add_generation_prompt=add_generation_prompt)


class TiktokenWrapper(TokenizerInterface):
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path):
        super().__init__(model_path)
        assert os.path.isfile(model_path), str(model_path)
        mergeable_ranks = load_tiktoken_bpe(str(model_path))
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        # BOS / EOS token IDs
        self._bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self._eos_id: int = self.special_tokens["<|end_of_text|>"]

    def encode(self, text):
        return self.model.encode(text)

    def decode(self, tokens):
        return self.model.decode(tokens)

    def bos_id(self):
        return self._bos_id

    def eos_id(self):
        return self._eos_id

def get_tokenizer(tokenizer_model_path, model_name):
    """
    Factory function to get the appropriate tokenizer based on the model name.
    
    Args:
    - tokenizer_model_path (str): The file path to the tokenizer model.
    - model_name (str): The name of the model, used to determine the tokenizer type.

    Returns:
    - TokenizerInterface: An instance of a tokenizer.
    """

    if "llama-3" in str(model_name).lower():
        return TiktokenWrapper(tokenizer_model_path)
    elif "qwen" in str(model_name).lower():
        return HFJsonTokenizerWrapper(tokenizer_model_path)
    else:
        return SentencePieceWrapper(tokenizer_model_path)
