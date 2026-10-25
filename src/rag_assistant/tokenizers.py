"""Tokenizers for lexical retrieval.

- "word_stem": lowercase words of 2+ word characters (bm25s' default pattern), English stopwords removed, stemmed.
- "identifier_split": additionally splits identifiers on snake_case, camelCase, dots and slashes, so that
  "RecursiveCharacterTextSplitter" -> recursive, character, text, splitter. The full identifier is kept as an
  extra token, so exact identifier matches still score higher than matching only the parts.
"""

import re
from collections.abc import Callable
from functools import lru_cache

import Stemmer
from bm25s.stopwords import STOPWORDS_EN

_STOPWORDS = frozenset(STOPWORDS_EN)
_STEMMER = Stemmer.Stemmer("english")
_CACHE_SIZE = 2**20

_WORD_RE = re.compile(r"(?u)\b\w\w+\b")
# \w includes "_", so snake_case stays together here and is split in _identifier_terms; ".", "/" and "-" split here.
_IDENTIFIER_RE = re.compile(r"\w+")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


@lru_cache(maxsize=_CACHE_SIZE)
def _word_terms(word: str) -> tuple[str, ...]:
    """Return the term of one word: lowercased and stemmed, or nothing for a stopword (cached per word).

    Args:
        word: One word as matched by _WORD_RE.

    Returns:
        A tuple with the stemmed word, or an empty tuple for a stopword. A tuple, because the cached result is shared.
    """
    word = word.lower()
    return () if word in _STOPWORDS else (_STEMMER.stemWord(word),)


@lru_cache(maxsize=_CACHE_SIZE)
def _identifier_terms(word: str) -> tuple[str, ...]:
    """Return the terms of one word, split on snake_case and camelCase (cached per word).

    Args:
        word: One word as matched by _IDENTIFIER_RE.

    Returns:
        The stemmed parts, plus the whole word if it was split; parts shorter than 2 characters and stopwords are
        dropped.
    """
    parts = [p.lower() for chunk in word.split("_") for p in (_CAMEL_RE.findall(chunk) or [chunk])]
    if len(parts) > 1:
        parts.append(word.lower())
    return tuple(_STEMMER.stemWord(p) for p in parts if len(p) >= 2 and p not in _STOPWORDS)


def tokenize_word_stem(text: str) -> list[str]:
    """Tokenize text into lowercase, stemmed words without stopwords.

    Args:
        text: The text to tokenize (a corpus chunk or a query).

    Returns:
        The terms, in text order.
    """
    return [t for word in _WORD_RE.findall(text) for t in _word_terms(word)]


def tokenize_identifier_split(text: str) -> list[str]:
    """Tokenize text like tokenize_word_stem, but also split code identifiers into their parts.

    Args:
        text: The text to tokenize (a corpus chunk or a query).

    Returns:
        The terms, in text order.
    """
    return [t for word in _IDENTIFIER_RE.findall(text) for t in _identifier_terms(word)]


TOKENIZERS: dict[str, Callable[[str], list[str]]] = {
    "word_stem": tokenize_word_stem,
    "identifier_split": tokenize_identifier_split,
}


def get_tokenizer(name: str) -> Callable[[str], list[str]]:
    """Return the tokenizer function for a name.

    Args:
        name: A key of TOKENIZERS, "word_stem" or "identifier_split".

    Returns:
        The tokenizer function.

    Raises:
        ValueError: If the name is unknown.
    """
    if name not in TOKENIZERS:
        raise ValueError(f"Unknown tokenizer {name!r}, choose from {list(TOKENIZERS)}")
    return TOKENIZERS[name]
