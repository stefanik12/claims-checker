from typing import List

import numpy as np


class Token:

    def __init__(self, token: str, embedding: np.array = None, tags: List[str] = None):
        self.text = token
        self.embedding = embedding
        self.tag = tags


class TokenNode(Token):

    def __init__(self, token: str, embedding: np.array, tags: str,
                 parents: Token = None, ancestors: List[Token] = None):
        super().__init__(token, embedding, tags)

        self.parents = parents
        self.ancestors = ancestors
