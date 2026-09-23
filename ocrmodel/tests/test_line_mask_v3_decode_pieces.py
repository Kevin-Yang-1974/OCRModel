"""The per-token decode must reconstruct the full decode exactly.

``_token_annotations`` attributes each generated character to the token that
produced it by concatenating per-token decodes and asserting the result equals
the prediction.  Getting that decomposition wrong does not raise -- it just makes
``token_character_mapping_reliable`` false on every page, which zeroes
``alignment_coverage`` and every line-IoU statistic.  That is exactly what
happened on the first v3 attempt (0 of 104 pages mapped), so the property is
worth a test that fails loudly.

The failing case is a byte-level BPE tokenizer: one Chinese character spans
several tokens, and decoding any one of them alone yields U+FFFD rather than the
character's first bytes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "evaluation"))

from diagnose_line_mask_v3 import decode_pieces


class ByteLevelTokenizer:
    """A stand-in for GLM-OCR's byte-level tokenizer.

    Each token is one UTF-8 byte, so a multi-byte character only decodes once
    every one of its bytes has been seen.  An incomplete tail decodes to U+FFFD,
    which is the behaviour that broke the original implementation.
    """

    def __init__(self, text: str):
        self.bytes = [bytes([byte]) for byte in text.encode("utf-8")]

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        raw = b"".join(self.bytes[int(i)] for i in ids)
        return raw.decode("utf-8", errors="replace")


def _chars(text: str) -> str:
    return "".join(char for char in text if not char.isspace())


def test_pieces_reproduce_a_multi_byte_decode_exactly():
    text = "尼佛舍利塔裝以綠色琉璃瓦"
    tokenizer = ByteLevelTokenizer(text)
    ids = list(range(len(tokenizer.bytes)))

    pieces = decode_pieces(tokenizer, ids)

    assert "".join(pieces) == tokenizer.decode(ids) == text
    assert "".join(pieces) == text


def test_each_character_is_attributed_to_the_token_that_completes_it():
    """A character assembled from several tokens belongs to the last of them."""

    text = "甲乙"           # 3 bytes each under UTF-8
    tokenizer = ByteLevelTokenizer(text)
    ids = list(range(len(tokenizer.bytes)))

    pieces = decode_pieces(tokenizer, ids)

    assert pieces[:2] == ["", ""]        # first two bytes of 甲 resolve nothing yet
    assert pieces[2] == "甲"             # the byte that completes it owns it
    assert pieces[3:5] == ["", ""]
    assert pieces[5] == "乙"


def test_no_replacement_characters_leak_into_the_pieces():
    """The U+FFFD of an incomplete tail must never reach a piece."""

    text = "須菩提如恒河中所有沙數"
    tokenizer = ByteLevelTokenizer(text)
    ids = list(range(len(tokenizer.bytes)))

    pieces = decode_pieces(tokenizer, ids)

    assert "�" not in "".join(pieces)
    assert _chars("".join(pieces)) == _chars(text)


def test_the_old_isolation_method_is_what_fails():
    """Documents the bug: per-token decode cannot reproduce the text."""

    text = "尼佛舍利塔"
    tokenizer = ByteLevelTokenizer(text)
    ids = list(range(len(tokenizer.bytes)))

    isolated = "".join(tokenizer.decode([i]) for i in ids)

    assert isolated != text
    assert "�" in isolated
    # ...while the supported method is exact, for the same ids.
    assert "".join(decode_pieces(tokenizer, ids)) == text


def test_an_empty_token_list_produces_no_pieces():
    tokenizer = ByteLevelTokenizer("甲")
    assert decode_pieces(tokenizer, []) == []
