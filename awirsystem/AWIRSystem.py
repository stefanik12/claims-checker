from typing import Tuple, List
from scipy.spatial import distance
import numpy as np

from pv211_utils.irsystem import IRSystem
from pv211_utils.loader import load_documents

from embedding import TransformerEmbedding
from nn_wrappers import AutoTransformerModule
from objects import Document
from weighting import SelectedHeadsAttentionWeighting


class AWIRSystem(IRSystem):
    """
    1. On init, weight documents parts by selected attention aggregation strategy
    2. On search
        2.1 Match candidate documents by contextual embeddings of doc tokens to query tokens
        2.2 Rank the candidate documents by the attention weight of their matching parts
    """

    def __init__(self, transformer_name_or_path: str = "bert-base-uncased"):
        # weight documents, index weighting and tokens by their embeddings, to use it then in search
        transformer = AutoTransformerModule(transformer_name_or_path)
        self.weighter = SelectedHeadsAttentionWeighting(transformer,
                                                        heads_subset=self._subset_heads(),
                                                        aggregation_strategy="per-layer-mean+sum")
        self.embeder = TransformerEmbedding(transformer, embed_strategy="last-4")
        self.inv_embedding_index = dict()
        self.doc_index = set()
        for doc_id, doc in load_documents(Document):
            tokens_a, weights = self.weighter.tokenize_weight_text(doc.body)
            tokens_e, embeddings = self.embeder.tokenize_embed_text(doc.body)
            assert len(tokens_a) == len(tokens_e)

            for token, embedding, weight in zip(tokens_a, embeddings, weights):
                self.inv_embedding_index[embedding] = (weight, token, doc_id)
                self.doc_index.add(doc_id)

    def _eval_match(self, matches: List[Tuple[int, str]], weight_embedding_ratio: float) -> float:
        return 0

    def search(self, query: str, weight_query: bool = False, distance_f: str = "cosine",
               match_topn: int = 10, retrieve_topn: int = 10, weight_embedding_ratio: float = 1.0) -> List[str]:
        """The ranked retrieval results for a query.

        Parameters
        ----------
            :param query : input query
            :param weight_query: Whether to weight matches of query tokens by their inner weights
            :param distance_f:
            :param match_topn:
            :param retrieve_topn:
            :param weight_embedding_ratio:

        Returns
        -------
        list of Document
            The ranked retrieval results for a query.
        """
        embedding_idx = np.ndarray(self.inv_embedding_index.keys())
        if match_topn is None or not match_topn:
            match_topn = len(embedding_idx)

        docs_scores = {doc_id: [] for doc_id in self.doc_index}

        for token, embedding in self.embeder.tokenize_embed_text(query):
            distances = distance.cdist(np.array([embedding]), embedding_idx, distance_f)[0]
            dist_e_rank = sorted(list(zip(distances, embedding_idx)), key=lambda d_e: d_e[0])
            for dist, response_token_e in dist_e_rank[:match_topn]:
                match_weight, match_token, match_doc_id = self.inv_embedding_index[response_token_e]
                docs_scores[match_doc_id].append((match_weight, match_token))
        # TODO: aggregate per-document scores by selected strategy
        # TODO: we are minimizing weights, but maximizing rank
        docs_ranked = sorted(docs_scores.items(), key=lambda docid_docm: self._eval_match(docid_docm[1],
                                                                                          weight_embedding_ratio))
        print(docs_ranked)
        docs_returned = [doc_id for doc_id, doc_score in docs_ranked[:retrieve_topn]]

        return docs_returned

    def _subset_heads(self) -> List[Tuple[int, int]]:
        from heads_idx import best_to_worst_heads
        return best_to_worst_heads
