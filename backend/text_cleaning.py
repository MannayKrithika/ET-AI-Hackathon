"""
Shared text cleaning — fixes encoding artifacts that can show up in
industrial documents/spreadsheets (mojibake, stray script fragments from
upstream data corruption). Used by both extraction.py and chunking.py so a
document is cleaned the same way regardless of which stage touches it.
"""

import re
import ftfy

# Matches CJK / Hangul / Kana script runs. This dataset is English-language
# industrial documentation, so a script fragment like "cavitation损害" glued
# onto an English word is upstream data corruption, not intentional
# multilingual content — strip it rather than let it into embeddings.
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\u30a0-\u30ff\uac00-\ud7af]+")


def clean_text_encoding(text) -> str:
    """
    Fixes genuine mojibake (e.g. UTF-8 bytes misread as Latin-1) via ftfy,
    then strips stray CJK fragments that show up glued onto otherwise-English
    text. Safe to call on any input; None/non-string collapses to "".
    """
    if text is None:
        return ""
    text = str(text)
    if not text.strip():
        return text
    fixed = ftfy.fix_text(text)
    fixed = _CJK_PATTERN.sub("", fixed)
    fixed = re.sub(r"[ \t]+", " ", fixed).strip()
    return fixed
