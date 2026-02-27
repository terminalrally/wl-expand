import os
import sys
from typing import List, Set, Optional

from wl_expand.Models import EmbedModel, Transformer
from wl_expand.EmbeddingEngine import EmbeddingEngine
from wl_expand.TransformerEngine import TransformerEngine
from wl_expand.MutationEngine import MutationEngine


class WordlistExpander:
    def __init__(
        self,
        embed_model: EmbedModel = EmbedModel.DEFAULT,
        transformer_model: Optional[Transformer] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
        num_words: int = 10,
        case_sensitive: bool = False,
        verbose: bool = False,
        workers: int = 0,
        mutate: bool = False,
    ):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.num_words = num_words
        self.case_sensitive = case_sensitive
        self.verbose = verbose
        self.workers = workers if workers > 0 else os.cpu_count() or 1
        self.mutate = mutate

        self.embed_engine = EmbeddingEngine(model_type=embed_model, verbose=verbose, workers=self.workers)
        self.mutation_engine = MutationEngine(case_sensitive=case_sensitive)

        self.transformer_engine: Optional[TransformerEngine] = None
        if transformer_model is not None:
            self.transformer_engine = TransformerEngine(model_type=transformer_model, verbose=verbose)

    def load_models(self) -> None:
        self.embed_engine.load()
        if self.transformer_engine is not None:
            self.transformer_engine.load()

    def expand_word(self, word: str) -> List[str]:
        if self.verbose:
            print(f"\n[expand] Seed word: '{word}'", file=sys.stderr)

        candidates = self.embed_engine.similar(
            word,
            top_k=self.top_k,
            threshold=self.similarity_threshold,
        )

        if self.verbose:
            print(f"[expand] Embedding candidates ({len(candidates)}):", file=sys.stderr)
            for w, s in candidates:
                print(f"  {w:30s}  {s:.4f}", file=sys.stderr)

        if self.transformer_engine is not None and candidates:
            candidates = self.transformer_engine.rerank(word, candidates, weight=0.3)
            if self.verbose:
                print(f"[expand] After transformer re-rank:", file=sys.stderr)
                for w, s in candidates:
                    print(f"  {w:30s}  {s:.4f}", file=sys.stderr)

        semantic_words = [w for w, _ in candidates]

        all_base_words = [word] + semantic_words
        result: Set[str] = set(all_base_words)

        if self.mutate:
            for base in all_base_words:
                mutations = self.mutation_engine.mutate(base, max_variants=self.num_words)
                result.update(mutations)

        if self.verbose:
            print(f"[expand] Total unique variants: {len(result)}", file=sys.stderr)

        if not self.case_sensitive:
            result_list = sorted(result, key=str.lower)
        else:
            result_list = sorted(result)

        return result_list

    def expand_words(self, words: List[str]) -> List[str]:
        from concurrent.futures import ThreadPoolExecutor

        if len(words) <= 1:
            all_results: Set[str] = set()
            for word in words:
                all_results.update(self.expand_word(word))
        else:
            # Batch embedding lookup across all seeds at once
            batch_candidates = self.embed_engine.similar_batch(
                words,
                top_k=self.top_k,
                threshold=self.similarity_threshold,
            )

            # Parallel mutation across all seeds + neighbors
            all_results: Set[str] = set()

            def _mutate_word(word: str) -> List[str]:
                candidates = batch_candidates.get(word, [])

                # transformer re-ranking
                if self.transformer_engine is not None and candidates:
                    candidates = self.transformer_engine.rerank(word, candidates, weight=0.3)

                semantic_words = [w for w, _ in candidates]
                all_base_words = [word] + semantic_words

                result: Set[str] = set(all_base_words)
                if self.mutate:
                    for base in all_base_words:
                        mutations = self.mutation_engine.mutate(base, max_variants=self.num_words)
                        result.update(mutations)
                return list(result)

            if self.verbose:
                print(f"[expand] Mutating {len(words)} seeds using {self.workers} workers", file=sys.stderr)

            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                for expanded in pool.map(_mutate_word, words):
                    all_results.update(expanded)

        if not self.case_sensitive:
            return sorted(all_results, key=str.lower)
        return sorted(all_results)

    def expand_from_file(self, filepath: str) -> List[str]:
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Input file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            words = [line.strip() for line in f if line.strip()]

        if self.verbose:
            print(f"[expand] Read {len(words)} seed words from '{filepath}'", file=sys.stderr)

        return self.expand_words(words)
