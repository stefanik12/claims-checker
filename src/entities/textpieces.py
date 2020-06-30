from typing import Iterable, Collection, Union, List
import numpy as np

from src.entities.token import Token


class Sentence:

    def __init__(self, sample_i: int, context_i: int, tokens: Collection[Token]):
        self.sample_i = sample_i
        self.context_i = context_i
        self.tokens = tokens

    @staticmethod
    def from_text(text: str, sample_i: int = None, context_i: int = None,
                  embeddings: Collection[np.array] = None, tags: Collection[List[str]] = None):
        from nltk.tokenize import word_tokenize
        words = word_tokenize(text)
        if embeddings is not None:
            assert len(words) == len(embeddings)
        else:
            embeddings = [None for _ in words]
        if tags is not None:
            assert len(words) == len(tags)
        else:
            tags = [None for _ in words]
        return Sentence(sample_i, context_i, [Token(w, e, t) for w, e, t in zip(words, embeddings, tags)])


class Paragraph:

    def __init__(self, sentences: Iterable[Sentence], sample_i: int = None, context_i: int = None):
        self.sample_i = sample_i
        self.context_i = context_i
        self.sentences = sentences

    @staticmethod
    def from_text(text: str, sample_i: int = None, context_i: int = None):
        from nltk.tokenize import sent_tokenize
        return Paragraph([Sentence.from_text(sent_text) for sent_text in sent_tokenize(text)], sample_i, context_i)


class ClassifiedParagraph(Paragraph):

    def __init__(self, sentences: Iterable[Sentence], sample_i: int, context_i: int):
        super().__init__(sentences, sample_i, context_i)
