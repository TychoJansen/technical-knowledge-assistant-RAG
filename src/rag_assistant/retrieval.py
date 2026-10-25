"""BM25 retrieval on top of bm25s, with a configurable tokenizer and an on-disk index per tokenizer."""

import json
from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import bm25s
import numpy as np

from rag_assistant.config import INDEX_DIR
from rag_assistant.data import load_corpus
from rag_assistant.logger import get_logger
from rag_assistant.tokenizers import get_tokenizer

logger = get_logger(__name__)

_DOC_IDS_FILE = "doc_ids.json"
_META_FILE = "meta.json"


@dataclass(frozen=True)
class RetrievedDoc:
    """A single search hit. Rank is 1-based."""

    doc_id: str
    score: float
    rank: int


class Retriever(ABC):
    """A retriever maps a query string to a ranked list of corpus doc_ids.

    Implementations may only use the corpus and the query text.
    """

    @abstractmethod
    def search(self, query: str, k: int) -> list[RetrievedDoc]:
        """Return the top-k documents for a query, best first.

        Args:
            query: The search query.
            k: The number of top documents to return.

        Returns:
            A list of retrieved documents with their scores and ranks.
        """


class BM25Retriever(Retriever):
    """Lexical BM25 retriever (bm25s defaults: Lucene variant, k1=1.5, b=0.75)."""

    def __init__(self, model: bm25s.BM25, doc_ids: list[str], tokenizer: str) -> None:
        """Wrap a fitted bm25s model; doc_ids[i] is the id of the i-th indexed document.

        Args:
            model: A fitted bm25s.BM25 instance.
            doc_ids: A list of corpus document ids, in the same order as the indexed documents.
            tokenizer: The name of the tokenizer used to index the corpus.
        """
        self.model = model
        self.doc_ids = doc_ids
        self.tokenizer = tokenizer
        self._tokenize = get_tokenizer(tokenizer)

    @classmethod
    def build(cls, doc_ids: Sequence[str], texts: Sequence[str], tokenizer: str) -> "BM25Retriever":
        """Tokenize the corpus and fit a BM25 index.

        Args:
            doc_ids: A sequence of corpus document ids, in the same order as texts.
            texts: A sequence of corpus document texts, in the same order as doc_ids.
            tokenizer: The name of the tokenizer to use for indexing.

        Returns:
            A BM25Retriever instance with a fitted index.
        """
        tokenize = get_tokenizer(tokenizer)
        logger.info(f"Tokenizing {len(texts)} documents with the {tokenizer!r} tokenizer")
        corpus_tokens = [tokenize(text) for text in texts]
        model = bm25s.BM25()
        model.index(corpus_tokens, show_progress=False)
        logger.info(f"Indexed {len(doc_ids)} documents, vocabulary size {len(model.vocab_dict)}")
        return cls(model, list(doc_ids), tokenizer)

    def save(self, index_dir: Path) -> None:
        """Save the index, the doc_id mapping and the tokenizer name.

        Args:
            index_dir: The directory to save the index in. Will be created if it doesn't exist.
        """
        index_dir.mkdir(parents=True, exist_ok=True)
        # meta.json marks a complete index: remove it first and write it last, so a crashed save is never "complete".
        (index_dir / _META_FILE).unlink(missing_ok=True)
        self.model.save(index_dir, show_progress=False)
        (index_dir / _DOC_IDS_FILE).write_text(json.dumps(self.doc_ids), encoding="utf-8")
        (index_dir / _META_FILE).write_text(json.dumps({"tokenizer": self.tokenizer}), encoding="utf-8")

    @staticmethod
    def exists(index_dir: Path) -> bool:
        """Return whether index_dir contains a complete index saved with save().

        Args:
            index_dir: The directory to check.

        Returns:
            True if the index is complete (meta.json, written last by save(), is present).
        """
        return (index_dir / _META_FILE).exists()

    @classmethod
    def load(cls, index_dir: Path) -> "BM25Retriever":
        """Load an index saved with save().

        Args:
            index_dir: The directory from which to load the index.

        Returns:
            A BM25Retriever instance with the loaded index.

        Raises:
            FileNotFoundError: If the index_dir does not contain a saved index.
        """
        if not cls.exists(index_dir):
            raise FileNotFoundError(f"No BM25 index in {index_dir}; run `rag index` first")
        model = bm25s.BM25.load(index_dir, show_progress=False)
        doc_ids = json.loads((index_dir / _DOC_IDS_FILE).read_text(encoding="utf-8"))
        meta = json.loads((index_dir / _META_FILE).read_text(encoding="utf-8"))
        return cls(model, doc_ids, meta["tokenizer"])

    def search(self, query: str, k: int) -> list[RetrievedDoc]:
        """Return the top-k documents with a positive BM25 score.

        Args:
            query: The search query.
            k: The number of top documents to return.

        Returns:
            A list of retrieved documents with their scores and ranks; fewer than k if fewer documents share a term with
            the query, and empty if the query has no terms.
        """
        tokens = self._tokenize(query)
        if not tokens:
            logger.warning(f"Query {query!r} tokenized to empty list, returning no results")
            return []
        indices, scores = self.model.retrieve([tokens], k=min(k, len(self.doc_ids)), show_progress=False)
        hits = [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if s > 0]
        return [RetrievedDoc(self.doc_ids[i], s, rank) for rank, (i, s) in enumerate(hits, start=1)]

    def search_within(self, query: str, k: int, doc_ids: Collection[str]) -> list[RetrievedDoc]:
        """Return the top-k documents with a positive BM25 score, searching only among the given doc_ids.

        Used by the insufficient-context evaluation, which gives the model only chunks known not to answer the question
        (rag_assistant.evaluation.abstention_eval). Scores are the same as in search(); the other documents are masked.

        Args:
            query: The search query.
            k: The number of top documents to return.
            doc_ids: The documents to search among; ids that are not in the index are ignored.

        Returns:
            A list of retrieved documents with their scores and ranks.
        """
        tokens = self._tokenize(query)
        positions = [self._positions[d] for d in doc_ids if d in self._positions]
        if not tokens or not positions:
            return []
        mask = np.zeros(len(self.doc_ids), dtype=np.float32)
        mask[positions] = 1.0
        indices, scores = self.model.retrieve([tokens], k=min(k, len(positions)), show_progress=False, weight_mask=mask)
        hits = [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if s > 0]
        return [RetrievedDoc(self.doc_ids[i], s, rank) for rank, (i, s) in enumerate(hits, start=1)]

    @cached_property
    def _positions(self) -> dict[str, int]:
        """doc_id -> position in the index; built on first use (search_within only)."""
        return {d: i for i, d in enumerate(self.doc_ids)}


def index_dir(tokenizer: str) -> Path:
    """Return the directory of the BM25 index for a tokenizer.

    Args:
        tokenizer: The tokenizer name.

    Returns:
        The index directory, indexes/bm25_<tokenizer>.
    """
    return INDEX_DIR / f"bm25_{tokenizer}"


def index_exists(tokenizer: str) -> bool:
    """Return whether a complete saved BM25 index exists for this tokenizer.

    Args:
        tokenizer: The tokenizer name.

    Returns:
        True if the index directory holds a complete index.
    """
    return BM25Retriever.exists(index_dir(tokenizer))


def build_index(tokenizer: str) -> None:
    """Build the BM25 index from the local corpus and save it to disk.

    Args:
        tokenizer: The name of the tokenizer to use.
    """
    corpus = load_corpus(columns=["doc_id", "text"])
    bm25 = BM25Retriever.build(corpus["doc_id"].tolist(), corpus["text"].tolist(), tokenizer)
    bm25.save(index_dir(tokenizer))


def load_retriever(tokenizer: str) -> BM25Retriever:
    """Load the BM25 retriever from its saved index.

    Args:
        tokenizer: The tokenizer the index was built with.

    Returns:
        A loaded retriever.
    """
    return BM25Retriever.load(index_dir(tokenizer))
