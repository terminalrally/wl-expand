import itertools
from typing import List, Set


# Common leet-speak substitution maps
_LEET_MAP = {
    "a": ["@", "4"],
    "b": ["8"],
    "c": ["("],
    "e": ["3"],
    "g": ["6", "9"],
    "h": ["#"],
    "i": ["1", "!", "|"],
    "l": ["1"],
    "o": ["0"],
    "s": ["$", "5"],
    "t": ["7"],
    "z": ["2"],
}

# Common password suffixes
_COMMON_SUFFIXES = [
    "", "1", "12", "123", "1234", "12345",
    "!", "!!", "!!!", "@", "#", "$",
    "01", "02", "69", "99", "00",
    "24", "25", "67", # Gen-Z favorites
    "2010", "2011", "2012", "2013", "2014",
    "2015", "2016", "2017", "2018", "2019",
    "2020", "2021", " 2022", "2023", "2024",
    "2025", "2026",
]

# Common password prefixes
_COMMON_PREFIXES = [
    "", "!", "@", "#", "1", "the", "my",
]

# QWERTY Keyboard-adjacent character map
# TODO add support for alternate layouts
_KEYBOARD_ADJACENT = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "erfcxs",
    "e": "rdsw3", "f": "rtgvcd", "g": "tyhbvf", "h": "yujnbg",
    "i": "ujko8", "j": "uikmnh", "k": "iolmj", "l": "opk",
    "m": "njk", "n": "bhjm", "o": "iklp9", "p": "ol0",
    "q": "wa12", "r": "edft4", "s": "wedxza", "t": "rfgy5",
    "u": "yhji7", "v": "cfgb", "w": "qase2", "x": "zsdc",
    "y": "tghu6", "z": "asx",
}


class MutationEngine:
    def __init__(self, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive

    def mutate(self, word: str, max_variants: int = 100) -> List[str]:
        variants: Set[str] = set()

        variants.update(self._case_variants(word))
        variants.update(self._leet_variants(word))
        variants.update(self._suffix_variants(word))
        variants.update(self._prefix_variants(word))
        variants.update(self._keyboard_typos(word))
        
        variants.discard(word) # remove original

        result = sorted(variants)
        return result[:max_variants]

    def _case_variants(self, word: str) -> List[str]:
        variants = [
            word.lower(),
            word.upper(),
            word.capitalize(),
            word.swapcase(),
            word.lower().capitalize(),
        ]
        # Also toggle first char
        if len(word) > 0:
            variants.append(word[0].swapcase() + word[1:])
        return variants

    def _leet_variants(self, word: str, max_positions: int = 3) -> List[str]:
        lower = word.lower()
        # Determine where leet substitutions are possible
        leet_positions = []
        for i, ch in enumerate(lower):
            if ch in _LEET_MAP:
                leet_positions.append(i)

        if not leet_positions:
            return []

        variants: List[str] = []

        # Single-char substitutions
        for pos in leet_positions:
            ch = lower[pos]
            for replacement in _LEET_MAP[ch]:
                variant = lower[:pos] + replacement + lower[pos + 1:]
                variants.append(variant)

        # Multi-char substitutions
        # Only up to `max_positions` positions to limit explosion
        for r in range(2, min(max_positions + 1, len(leet_positions) + 1)):
            for combo in itertools.combinations(leet_positions, r):
                # pick one replacement per position
                replacement_options = [_LEET_MAP[lower[pos]] for pos in combo]
                for replacements in itertools.product(*replacement_options):
                    chars = list(lower)
                    for pos, rep in zip(combo, replacements):
                        chars[pos] = rep
                    variants.append("".join(chars))
                    if len(variants) > 500:
                        return variants

        return variants

    def _suffix_variants(self, word: str) -> List[str]:
        return [word + suffix for suffix in _COMMON_SUFFIXES if suffix]

    def _prefix_variants(self, word: str) -> List[str]:
        return [prefix + word for prefix in _COMMON_PREFIXES if prefix]

    def _keyboard_typos(self, word: str) -> List[str]:
        lower = word.lower()
        variants: List[str] = []

        for i, ch in enumerate(lower):
            adjacent = _KEYBOARD_ADJACENT.get(ch, "")
            for adj_ch in adjacent:
                variant = lower[:i] + adj_ch + lower[i + 1:]
                variants.append(variant)

        return variants
