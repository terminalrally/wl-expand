import os
import sys
from typing import List, Tuple, Optional

from wl_expand.Models import Transformer


# Map enum values to HuggingFace model identifiers
_TRANSFORMER_MAP = {
    Transformer.MINILM_L3_V2: "sentence-transformers/paraphrase-MiniLM-L3-v2",
    Transformer.MINILM_L6_V2: "sentence-transformers/all-MiniLM-L6-v2",
    Transformer.MPNET_BASE_V2: "sentence-transformers/all-mpnet-base-v2",
}

# Local cache directory for saved transformer models
_CACHE_DIR = os.path.join(os.path.expanduser("~"), "gensim-data", "transformers")


class TransformerEngine:
    def __init__(self, model_type: Transformer = Transformer.DEFAULT, verbose: bool = False):
        self.model_type = model_type
        self.verbose = verbose
        self.model = None

    @staticmethod
    def _cache_path(model_type: Transformer) -> str:
        return os.path.join(_CACHE_DIR, model_type.value)

    def load(self) -> None:
        from sentence_transformers import SentenceTransformer

        cache_path = self._cache_path(self.model_type)
        model_name = _TRANSFORMER_MAP[self.model_type]

        if os.path.exists(cache_path):
            if self.verbose:
                print(f"[transformer] Loading cached model: {cache_path} ...", file=sys.stderr)
            self.model = SentenceTransformer(cache_path)
        else:
            if self.verbose:
                print(f"[transformer] First run - downloading {model_name} ...", file=sys.stderr)
                print(f"[transformer] (this is slow once; subsequent loads will be instant)", file=sys.stderr)
            self.model = SentenceTransformer(model_name)
            os.makedirs(cache_path, exist_ok=True)
            self.model.save(cache_path)
            if self.verbose:
                print(f"[transformer] Saved to cache: {cache_path}", file=sys.stderr)

        if self.verbose:
            print(f"[transformer] Model loaded.", file=sys.stderr)

    def rerank(
        self,
        seed_word: str,
        candidates: List[Tuple[str, float]],
        weight: float = 0.5,
    ) -> List[Tuple[str, float]]:
        from sentence_transformers import util

        if self.model is None:
            raise RuntimeError("Model not loaded.")
        if not candidates:
            return []

        words = [w for w, _ in candidates]
        original_scores = [s for _, s in candidates]

        seed_embedding = self.model.encode(seed_word, convert_to_tensor=True)
        candidate_embeddings = self.model.encode(words, convert_to_tensor=True)
        cosine_scores = util.cos_sim(seed_embedding, candidate_embeddings)[0].cpu().numpy()

        blended = []
        for i, (word, orig_score) in enumerate(candidates):
            transformer_score = float(cosine_scores[i])
            final_score = (1.0 - weight) * orig_score + weight * transformer_score
            blended.append((word, final_score))

        blended.sort(key=lambda x: x[1], reverse=True)
        return blended

    def similarity(self, text1: str, text2: str) -> float:
        from sentence_transformers import util

        if self.model is None:
            raise RuntimeError("Model not loaded.")
        embeddings = self.model.encode([text1, text2], convert_to_tensor=True)
        score = util.cos_sim(embeddings[0], embeddings[1])
        return float(score[0][0])
