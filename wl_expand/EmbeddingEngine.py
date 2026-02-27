import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional

from wl_expand.Models import EmbedModel


# Map enum values to gensim model identifiers
_MODEL_MAP = {
    EmbedModel.FASTTEXT: "fasttext-wiki-news-subwords-300",
    EmbedModel.WORD2VEC: "word2vec-google-news-300",
    EmbedModel.GLOVE: "glove-wiki-gigaword-300",
}


class EmbeddingEngine:
    def __init__(self, model_type: EmbedModel = EmbedModel.DEFAULT, verbose: bool = False, workers: int = 0):
        self.model_type = model_type
        self.verbose = verbose
        self.model = None
        self.workers = workers if workers > 0 else os.cpu_count() or 1

        # Allow numpy/BLAS to use multiple cores for vector operations
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(self.workers))
        os.environ.setdefault("MKL_NUM_THREADS", str(self.workers))

    @staticmethod
    def _native_path(model_name: str) -> str:
        import gensim.downloader as api

        return os.path.join(api.BASE_DIR, model_name, f"{model_name}.kv")

    def load(self) -> None:
        import gensim.downloader as api
        from gensim.models import KeyedVectors

        model_name = _MODEL_MAP[self.model_type]
        native = self._native_path(model_name)

        if os.path.exists(native):
            if self.verbose:
                print(f"[embed] Loading native cache: {native} ...", file=sys.stderr)
            self.model = KeyedVectors.load(native, mmap="r")
        else:
            # First run: download gz model and convert to native format
            if self.verbose:
                print(f"[embed] First run - downloading and converting {model_name} ...", file=sys.stderr)
                print(f"[embed] (this is slow once; subsequent loads will be instant)", file=sys.stderr)
            kv = api.load(model_name)
            kv.save(native)
            if self.verbose:
                print(f"[embed] Saved native cache: {native}", file=sys.stderr)
            self.model = kv

        # Pre-compute and cache the unit-normalised vectors so that
        # most_similar() doesn't re-normalise on every call.
        self.model.fill_norms()

        if self.verbose:
            print(f"[embed] Loaded {len(self.model.key_to_index)} word vectors.", file=sys.stderr)

    def similar(self, word: str, top_k: int = 10, threshold: float = 0.0) -> List[Tuple[str, float]]:
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        try:
            # Fetch more candidates than needed so we can filter by threshold
            candidates = self.model.most_similar(word, topn=top_k * 3)
        except KeyError:
            if self.verbose:
                print(f"[embed] Word not in vocabulary: '{word}'", file=sys.stderr)
            return []

        results = [(w, score) for w, score in candidates if score >= threshold]
        return results[:top_k]

    def similarity(self, word1: str, word2: str) -> float:
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        try:
            return float(self.model.similarity(word1, word2))
        except KeyError:
            return 0.0

    def similar_batch(
        self, words: List[str], top_k: int = 10, threshold: float = 0.0
    ) -> Dict[str, List[Tuple[str, float]]]:
        if self.verbose:
            print(f"[embed] Batch lookup for {len(words)} words using {self.workers} workers", file=sys.stderr)

        def _lookup(word: str) -> Tuple[str, List[Tuple[str, float]]]:
            return word, self.similar(word, top_k=top_k, threshold=threshold)

        results: Dict[str, List[Tuple[str, float]]] = {}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for word, neighbors in pool.map(_lookup, words):
                results[word] = neighbors
        return results

    def in_vocab(self, word: str) -> bool:
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        return word in self.model.key_to_index
