from enum import Enum

class EmbedModel(Enum):
    DEFAULT = "fasttext"
    FASTTEXT = "fasttext"
    WORD2VEC = "word2vec"
    GLOVE = "glove"

class Transformer(Enum):
    DEFAULT = "minilm-l6-v2"
    MINILM_L3_V2 = "minilm-l3-v2"
    MINILM_L6_V2 = "minilm-l6-v2"
    MPNET_BASE_V2 = "mpnet-base-v2"