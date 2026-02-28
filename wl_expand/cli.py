#!/usr/bin/env python3
import argparse
import os
import sys

from wl_expand.Models import EmbedModel, Transformer
from wl_expand.WordlistExpander import WordlistExpander


def parse_filter(filter_str: str):
    filters = []
    if not filter_str:
        return filters

    # TODO maybe using luqum would be better here?
    for part in filter_str.split(","):
        part = part.strip()
        if part.startswith("length>"):
            min_len = int(part.split(">")[1])
            filters.append(lambda w, n=min_len: len(w) > n)
        elif part.startswith("length<"):
            max_len = int(part.split("<")[1])
            filters.append(lambda w, n=max_len: len(w) < n)
        elif part.startswith("length="):
            exact_len = int(part.split("=")[1])
            filters.append(lambda w, n=exact_len: len(w) == n)
        elif part.startswith("starts-with="):
            prefix = part.split("=", 1)[1]
            filters.append(lambda w, p=prefix: w.lower().startswith(p.lower()))
        elif part.startswith("ends-with="):
            suffix = part.split("=", 1)[1]
            filters.append(lambda w, s=suffix: w.lower().endswith(s.lower()))
        elif part.startswith("contains="):
            substr = part.split("=", 1)[1]
            filters.append(lambda w, s=substr: s.lower() in w.lower())
        elif part.startswith("excludes="):
            substr = part.split("=", 1)[1]
            filters.append(lambda w, s=substr: s.lower() not in w.lower())
    return filters


def apply_filters(words, filters):
    if not filters:
        return words
    return [w for w in words if all(f(w) for f in filters)]


def main():
    parser = argparse.ArgumentParser(
        description="Wordlist expansion using semantic similarity and language models",
        epilog=(
            "Examples:\n"
            "  %(prog)s password\n"
            "  %(prog)s -k 10 -s 0.6 -n 50 password admin login\n"
            "  %(prog)s -o expanded.txt seeds.txt\n"
            "  %(prog)s --embedding-model word2vec -f 'length>4' password\n"
            "  cat seeds.txt | %(prog)s\n"
            "  echo 'password' | %(prog)s --mutate\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("input", help="Input file or word(s) to expand (reads stdin if omitted)", nargs="*")

    parser.add_argument("-c", "--case-sensitive", help="Enable case-sensitive word expansion", action="store_true")
    parser.add_argument("-f", "--filter", help="Filter criteria for expanded words (e.g., 'length>5,starts-with=a')", default="")
    parser.add_argument("-m", "--mutate", help="Enable lexical mutations (leet-speak, typos, suffixes)", action="store_true")
    parser.add_argument("-n", "--num-words", help="Number of mutation variants per word", type=int, default=50)
    parser.add_argument("-o", "--output", help="Output file for expanded wordlist")
    parser.add_argument("-r", "--rerank", help="Enable sentence-transformer re-ranking of candidates", action="store_true")
    parser.add_argument("-s", "--similarity-threshold", help="Similarity threshold for word expansion (0.0 - 1.0, quality filter)", type=float, default=0.5)
    parser.add_argument("-k", "--top-k", help="Number of top similar words to consider for expansion (qty limit)", type=int, default=5)
    parser.add_argument("-v", "--verbose", help="Enable verbose output", action="store_true")
    parser.add_argument("-w", "--workers", help="Number of parallel workers (default: all CPU cores)", type=int, default=0)
    parser.add_argument("--embedding-model", help="Embedding model to use for similarity calculation", type=EmbedModel, default=EmbedModel.DEFAULT)
    parser.add_argument("--sentence-transformer", help="Sentence transformer model for re-ranking", type=Transformer, default=Transformer.DEFAULT)

    args = parser.parse_args()

    # Read from stdin if piped and no positional args given
    if not args.input and not sys.stdin.isatty():
        args.input = [line.strip() for line in sys.stdin if line.strip()]

    if not args.input:
        parser.print_usage()
        exit(1)

    # Load seed words
    seed_words = []
    input_files = []
    for item in args.input:
        if os.path.isfile(item):
            input_files.append(item)
        else:
            seed_words.append(item)

    expander = WordlistExpander(
        embed_model=args.embedding_model,
        transformer_model=args.sentence_transformer if args.rerank else None,
        top_k=args.top_k,
        similarity_threshold=args.similarity_threshold,
        num_words=args.num_words,
        case_sensitive=args.case_sensitive,
        verbose=args.verbose,
        workers=args.workers,
        mutate=args.mutate,
    )

    if args.verbose:
        print(f"[main] Embedding model : {args.embedding_model.value}", file=sys.stderr)
        if args.rerank:
            print(f"[main] Re-ranking      : {args.sentence_transformer.value}", file=sys.stderr)
        print(f"[main] Top-K           : {args.top_k}", file=sys.stderr)
        print(f"[main] Threshold       : {args.similarity_threshold}", file=sys.stderr)
        print(f"[main] Mutations/word  : {args.num_words}", file=sys.stderr)
        print(f"[main] Workers         : {expander.workers}", file=sys.stderr)
        print(f"[main] Case sensitive  : {args.case_sensitive}", file=sys.stderr)

    try:
        # Load models
        expander.load_models()
    except Exception as e:
        print(f"Error loading models: {e}", file=sys.stderr)
        sys.exit(1)

    results = []

    if seed_words:
        if args.verbose:
            print(f"[main] Expanding {len(seed_words)} seed word(s): {seed_words}", file=sys.stderr)
        results.extend(expander.expand_words(seed_words))

    for filepath in input_files:
        if args.verbose:
            print(f"[main] Expanding words from file: {filepath}", file=sys.stderr)
        results.extend(expander.expand_from_file(filepath))

    if args.case_sensitive:
        # Deduplication
        results = sorted(set(results))
    else:
        seen = set()
        unique = []
        for w in results:
            key = w.lower()
            if key not in seen:
                seen.add(key)
                unique.append(w)
        results = sorted(unique, key=str.lower)

    # Filtering
    filters = parse_filter(args.filter)
    results = apply_filters(results, filters)

    if args.verbose:
        print(f"[main] Final wordlist: {len(results)} words", file=sys.stderr)

    output_text = "\n".join(results) + "\n" if results else ""

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_text)
        if args.verbose:
            print(f"[main] Written to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(output_text)


if __name__ == "__main__":
    main()
