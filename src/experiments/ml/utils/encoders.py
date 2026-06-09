"""Text encoders for Machine Learning experiments.

Exposes four encoders with the same fit/transform interface:
- Bag of Words (CountVectorizer)
- TF-IDF (TfidfVectorizer)
- Word2Vec (gensim CBOW, trained on the corpus, mean-pooling)
- Sentence-Transformers (all-MiniLM-L6-v2, mean-pooling from the API itself)

Pre-trained dense encoders cache their output in `data/cache/` to avoid
recalculating expensive embeddings between reruns.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _hash_corpus(texts: Sequence[str]) -> str:
    """Stable hash of the corpus to identify cache entries."""
    h = hashlib.md5()
    for t in texts:
        h.update(t.encode("utf-8", errors="ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:12]


# ---------- Bag of Words ----------


class BoWEncoder:
    name = "BoW"
    output = "sparse"

    def __init__(self, ngram_range=(1, 2), min_df=2, max_features=50_000):
        from sklearn.feature_extraction.text import CountVectorizer

        self._vec = CountVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            lowercase=False,  # already normalized
        )

    def fit_transform(self, texts: Iterable[str]) -> Any:
        return self._vec.fit_transform(texts)

    def transform(self, texts: Iterable[str]) -> Any:
        return self._vec.transform(texts)


# ---------- TF-IDF ----------


class TfidfEncoder:
    name = "TF-IDF"
    output = "sparse"

    def __init__(self, ngram_range=(1, 2), min_df=2, max_features=50_000, sublinear_tf=True):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vec = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            lowercase=False,
        )

    def fit_transform(self, texts: Iterable[str]) -> Any:
        return self._vec.fit_transform(texts)

    def transform(self, texts: Iterable[str]) -> Any:
        return self._vec.transform(texts)


# ---------- Word2Vec (trained on corpus) ----------


class Word2VecEncoder:
    """Word2Vec (gensim, CBOW with `sg=0`) trained on the training split.

    It is trained dynamically exclusively on the training split (Train) 
    to avoid data leakage, being extraordinarily fast when processing thousands of SMS.
    """

    name = "Word2Vec"
    output = "dense"

    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 2, workers: int = 4, epochs: int = 5):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self._model = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.split()

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        from gensim.models import Word2Vec

        tokenized = [self._tokenize(t) for t in texts]
        # Gensim on Windows only supports one stable thread (spawn); avoids crashes or errors.
        workers = 1 if os.name == "nt" else self.workers
        self._model = Word2Vec(
            sentences=tokenized,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=workers,
            epochs=self.epochs,
            sg=0,  # CBOW (gensim)
        )
        return self._embed(tokenized)

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Word2VecEncoder.transform called before fit_transform")
        tokenized = [self._tokenize(t) for t in texts]
        return self._embed(tokenized)

    def _embed(self, tokenized: list[list[str]]) -> np.ndarray:
        assert self._model is not None, "Word2VecEncoder.fit_transform must be called before _embed"
        wv = self._model.wv
        dim = self.vector_size
        out = np.zeros((len(tokenized), dim), dtype=np.float32)
        for i, toks in enumerate(tokenized):
            vecs = [wv[t] for t in toks if t in wv]
            if vecs:
                out[i] = np.mean(vecs, axis=0)
        return out


# ---------- FastText (trained on corpus) ----------


class FastTextEncoder:
    """FastText (gensim) trained on the training split.
    
    It is trained dynamically exclusively on the training split.
    Similar to Word2Vec, it uses subword information (character n-grams),
    which greatly helps with rare or misspelled words typical of spam.
    """

    name = "FastText"
    output = "dense"

    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 2, workers: int = 4, epochs: int = 5):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self._model = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.split()

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        from gensim.models import FastText

        tokenized = [self._tokenize(t) for t in texts]
        # Gensim on Windows only supports one stable thread (spawn); avoids crashes or errors.
        workers = 1 if os.name == "nt" else self.workers
        self._model = FastText(
            sentences=tokenized,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=workers,
            epochs=self.epochs,
            sg=0,  # CBOW
        )
        return self._embed(tokenized)

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("FastTextEncoder.transform called before fit_transform")
        tokenized = [self._tokenize(t) for t in texts]
        return self._embed(tokenized)

    def _embed(self, tokenized: list[list[str]]) -> np.ndarray:
        assert self._model is not None, "FastTextEncoder.fit_transform must be called before _embed"
        wv = self._model.wv
        dim = self.vector_size
        out = np.zeros((len(tokenized), dim), dtype=np.float32)
        for i, toks in enumerate(tokenized):
            # In FastText, if the word is not present, n-grams are used,
            # but we can simply delegate to FastText's OOV handling.
            # We don't filter if the word is exactly in the vocab.
            vecs = []
            for t in toks:
                try:
                    vecs.append(wv[t])
                except KeyError:
                    pass
            if vecs:
                out[i] = np.mean(vecs, axis=0)
        return out


# ---------- Sentence-Transformers (MiniLM) ----------


class MiniLMEncoder:
    """Dense embeddings with `sentence-transformers/all-MiniLM-L6-v2`.

    Does not need to be trained: it is a pre-trained and frozen model. Therefore
    we can cache its embeddings on the full corpus of the dataset and
    reuse them efficiently (no data leakage).
    """

    name = "MiniLM"
    output = "dense"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self._cache_full: np.ndarray | None = None
        self._cache_index: dict[str, int] = {}

    def _build_cache(self, texts: Sequence[str], cache_key: str | None) -> None:
        """Encodes all texts once and builds a text->row index."""
        if cache_key is not None:
            cache_file = CACHE_DIR / f"minilm_{cache_key}.npy"
        else:
            cache_file = None

        if cache_file is not None and cache_file.exists():
            self._cache_full = np.load(cache_file)
        else:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.model_name)
            self._cache_full = model.encode(
                list(texts),
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            if cache_file is not None:
                np.save(cache_file, self._cache_full)

        self._cache_index = {}
        for i, t in enumerate(texts):
            self._cache_index.setdefault(t, i)

    def precompute(self, texts: Sequence[str], cache_key: str) -> None:
        """Pre-computes the full corpus of the dataset (idempotent with cache)."""
        if self._cache_full is None or len(self._cache_full) != len(texts):
            self._build_cache(texts, cache_key)

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        return self.transform(texts)

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self._cache_full is None:
            raise RuntimeError("Call precompute(...) before transform()")
        missing = [t for t in texts if t not in self._cache_index]
        if missing:
            ex = missing[0]
            ex_show = (ex[:80] + "…") if len(ex) > 80 else ex
            raise KeyError(
                "MiniLM: there are texts not present in the corpus passed to precompute(); "
                "the cache uses exact string matching. "
                f"Example ({len(missing)} not found): {ex_show!r}"
            )
        rows = [self._cache_index[t] for t in texts]
        return self._cache_full[rows]


# ---------- Registry ----------


def make_encoder(name: str) -> Any:
    """Instantiates and returns the requested encoder by name."""
    name = name.lower()
    if name == "bow":
        return BoWEncoder()
    if name == "tfidf":
        return TfidfEncoder()
    if name in ("w2v", "word2vec"):
        return Word2VecEncoder()
    if name in ("fasttext", "ft"):
        return FastTextEncoder()
    if name in ("minilm", "st", "sbert"):
        return MiniLMEncoder()
    raise ValueError(f"Unknown encoder: {name}")


ENCODER_NAMES = ["bow", "tfidf", "w2v", "fasttext", "minilm"]
