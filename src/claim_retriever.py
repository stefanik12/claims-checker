from typing import Union, Iterable

import torch
from torch.utils.data import DataLoader

from src.embedder.model import EmbeddingModel
from src.entities.frames import NounFrame
from src.entities.textpieces import Paragraph, Sentence, ClassifiedParagraph
from src.tagger.pos_tagger import AllenPosTagger


class NounFrameClaimRetriever(torch.nn.Module):
    frames = dict()
    index = dict()
    contexts = dict()

    def __init__(self, embedding_model: EmbeddingModel = None):
        super().__init__()
        self.embedding_model = embedding_model
        self.pos_tagger = AllenPosTagger()

    def add(self, paragraph: Paragraph):
        for sent in paragraph.sentences:
            sent.embeddings = self.embedding_model.embed_sentence(sent)
        for frame in self.pos_tagger.get_paragraph_frames(paragraph):
            self.frames[paragraph.sample_i] = frame
            self.index[paragraph.sample_i] = frame.noun.embedding
            self.contexts[paragraph.sample_i] = paragraph.context_i

    def remove(self, id_or_sample: Union[int, Paragraph]):
        if isinstance(id_or_sample, Paragraph):
            sample_i = id_or_sample.sample_i
        else:
            sample_i = id_or_sample
        if sample_i not in self.frames.keys():
            raise ValueError("Sample with id %s not in the index" % sample_i)
        del self.frames[sample_i]
        del self.index[sample_i]
        del self.contexts[sample_i]

    def fit(self, samples: Iterable[ClassifiedParagraph]):
        """
        Minimizes the distance of the indexed embeddings of the same meaning,
        i.e. residing in the paragraphs of the same id
        :param samples:
        :return:
        """

        pass

    def forward(self, dataloader: DataLoader):
        pass

    def retrieve(self, request: Union[str, NounFrame, Sentence, Iterable[Sentence]]):
        """
        Find the closest frame by indexed embedding and retrieve its content
        :param request:
        :return:
        """
        pass

