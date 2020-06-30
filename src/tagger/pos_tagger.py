from typing import Collection, Dict, Union, List, Iterable

import torch
from collections import deque
from src.config import CONFIG
from src.entities.frames import NounFrame
from src.entities.textpieces import Sentence, Paragraph

from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction


from src.entities.token import Token

pivot_tag = "NN"
minor_tag = "V"
minor_depth = 2


class AllenPosTagger(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tagger = Predictor.from_path(CONFIG["pos_tagger_url"])

    def tag_sentence_get_dependencies(self, sentence: Sentence) -> Dict[str, Union[str, List]]:
        dependencies_out = self.tagger.predict(sentence=" ".join([t.text for t in sentence.tokens]))
        sentence.tags = dependencies_out['pos']
        assert len(sentence.tags) == len(sentence.tokens)
        return dependencies_out['hierplane_tree']['root']

    def get_paragraph_frames(self, paragraph: Paragraph):
        return self.get_noun_frames(paragraph.sentences, paragraph.sample_i, paragraph.context_i)

    def get_noun_frames(self, sentences: Iterable[Sentence], paragraph_i: int, context_i: int) -> Iterable[NounFrame]:
        """
        Find the closest verb to the pivoting NN, and return that as a NounFrame.
        Root is a tree, thus we traverse it in BFS and assign the first verb to the first unassigned noun
        """
        for s_i, sentence in enumerate(sentences):
            depends_root = self.tag_sentence_get_dependencies(sentence)
            bfs_q = deque([depends_root])
            noun_q = deque()
            verb_q = deque()
            # the constructed noun frames must always have a pivot noun and a verb
            while len(bfs_q) > 0:
                w = bfs_q.popleft()
                if pivot_tag in w['attributes']:
                    # w is noun
                    try:
                        v = verb_q.pop()
                        yield NounFrame(Token(w["word"], tags=w["attributes"]),
                                        Token(v["word"], tags=v["attributes"]),
                                        paragraph_i, context_i)
                    except IndexError:
                        noun_q.append(w["word"])
                if any((a.startswith(minor_tag) for a in w['attributes'])):
                    # w is verb
                    try:
                        n = noun_q.pop()
                        yield NounFrame(Token(n["word"], tags=n["attributes"]),
                                        Token(w["word"], tags=w["attributes"]),
                                        paragraph_i, context_i)
                    except IndexError:
                        verb_q.append(w["word"])

                if "children" in w.keys():
                    bfs_q += w["children"]
