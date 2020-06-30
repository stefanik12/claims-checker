import abc

from src.entities.textpieces import Sentence


class EmbeddingModel(abc.ABC):

    @abc.abstractmethod
    def embed_sentence(self, sentence: Sentence):
        pass


class DummyEmbeddingModel(EmbeddingModel):

    def embed_sentence(self, sentence: Sentence):
        for token in sentence.tokens:
            token.embedding = len(token.text)
        return sentence
