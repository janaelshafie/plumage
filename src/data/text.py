"""Caption splitting and cleaning for the CUB ``description`` field.

Each row's ``description`` holds 10 captions joined by ``\\n``. Milestone 2
captioning reuses these helpers, so the vocabulary is built from the same text.

    >>> clean_captions("This bird is RED .\\n\\nIt has a short,  pointy beak!\\n")
    ['this bird is red', 'it has a short, pointy beak']
"""

from __future__ import annotations

import re
import unicodedata

# Typographic characters that appear in the raw captions, mapped to plain ASCII.
_CHAR_MAP = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", " ": " ",
})
_ALLOWED = re.compile(r"[^a-z0-9 ,.'\-]")      # everything else becomes a space
_REPEATED_PUNCT = re.compile(r"([,.'\-])\1+")   # "..", ",," -> single
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.])")
_MISSING_SPACE = re.compile(r"([,.])(?=[a-z])")  # "red,with" -> "red, with"
_WHITESPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def split_description(description: str) -> list[str]:
    """Split a raw description block into its captions, dropping blank lines."""
    return [line.strip() for line in description.split("\n") if line.strip()]


def clean_caption(caption: str) -> str:
    """Lowercase and normalize one caption.

    Steps: unicode NFKC, typographic quotes/dashes to ASCII, lowercase, drop
    characters outside ``[a-z0-9 ,.'-]``, collapse repeated punctuation and
    whitespace, fix spacing around commas/periods, strip trailing punctuation.
    Returns ``""`` if nothing is left.
    """
    text = unicodedata.normalize("NFKC", caption).translate(_CHAR_MAP).lower()
    text = _ALLOWED.sub(" ", text)
    text = _REPEATED_PUNCT.sub(r"\1", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _MISSING_SPACE.sub(r"\1 ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip(" ,.'-")


def clean_captions(description: str) -> list[str]:
    """``split_description`` + ``clean_caption``, dropping captions that end up empty."""
    cleaned = (clean_caption(c) for c in split_description(description))
    return [c for c in cleaned if c]


def tokenize(caption: str) -> list[str]:
    """Word tokens of a cleaned caption (punctuation dropped, apostrophes kept)."""
    return _TOKEN.findall(caption)
