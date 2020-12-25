from typing import Tuple, List

from pv211_utils.entities import QueryBase
from scipy.spatial import distance
import numpy as np

from pv211_utils.irsystem import IRSystem
from pv211_utils.loader import load_documents
from tqdm import tqdm
import torch

import config
from embedding import TransformerEmbedding
from nn_wrappers import AutoTransformerModule
from objects import Document, Match
from weighting import SelectedHeadsAttentionWeighting


class AWIRSystem(IRSystem):
    """
    1. On init, weight documents parts by selected attention aggregation strategy
    2. On search
        2.1 Match candidate documents by contextual embeddings of doc tokens to query tokens
        2.2 Rank the candidate documents by the attention weight of their matching parts
    """
    search_bar = None

    def __init__(self, transformer_name_or_path: str = "bert-base-uncased",
                 device: str = "cuda", infer_device: str = "cuda"):
        # weight documents, index weighting and tokens by their embeddings, to use it then in search
        transformer = AutoTransformerModule(transformer_name_or_path, device=device)
        self.weighter = SelectedHeadsAttentionWeighting(transformer,
                                                        heads_subset=self._subset_heads(),
                                                        aggregation_strategy="per-layer-mean+sum")
        self.embeder = TransformerEmbedding(transformer, embed_strategy="last-4")
        self.inv_embedding_index = []
        embedding_idx = []
        self.doc_index = set()
        for doc_id, doc in tqdm(list(load_documents(Document).items()), desc="Indexing"):
            tokens_a, weights = self.weighter.tokenize_weight_text(doc.body)
            tokens_e, embeddings = self.embeder.tokenize_embed_text(doc.body)
            assert len(tokens_a) == len(tokens_e)

            for token, embedding, weight in zip(tokens_a, embeddings, weights):
                embedding_idx.append(torch.tensor(embedding, dtype=torch.float32))
                self.inv_embedding_index.append((weight, token, doc_id))
                self.doc_index.add(doc_id)
        self.embedding_idx = torch.stack(embedding_idx, dim=0).to(infer_device)
        norms = torch.linalg.norm(self.embedding_idx, dim=1, ord=2).unsqueeze(-1)
        self.embedding_idx = self.embedding_idx / torch.max(norms, 1e-8 * torch.ones_like(norms))

    def search(self, query: QueryBase, weight_query: bool = False) -> List[str]:
        """The ranked retrieval results for a query.

        Parameters
        ----------
            :param query : input query
            :param weight_query: Whether to weight matches of query tokens by their inner attention weights
            :param distance_f:
            :param match_topn:
            :param retrieve_topn:

        Returns
        -------
        list of Document
            The ranked retrieval results for a query.
        """
        if self.search_bar is None:
            self.search_bar = tqdm(total=225)
        # print("Searching: %s" % query.body)

        if config.embedding_matches is None or not config.embedding_matches:
            config.embedding_matches = len(self.embedding_idx)

        matches = []

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        for token, embedding_l in zip(*self.embeder.tokenize_embed_text(query.body)):
            embedding = torch.tensor(embedding_l, dtype=torch.float32).to(self.embedding_idx.device)
            norm = torch.linalg.norm(embedding, ord=2)
            embedding_norm = embedding / torch.max(norm, 1e-10 * torch.ones_like(norm))

            distances = torch.matmul(self.embedding_idx, embedding_norm)
            for dist, idx in zip(*torch.sort(distances, descending=True)):
                attention, match_token, match_doc_id = self.inv_embedding_index[idx.item()]
                matches.append(Match(match_doc_id, match_token, dist.item(), attention))

        # done: aggregate per-document scores by selected)strategy
        # done: we are minimizing weights, but maximizing rank
        # done: all attention weights are the same, find out why
        # done: also check embeddings, all distances are the same now
        # debug:
        # docs = load_documents(Document)
        # [(m.weight, docs[m.doc_id]) for m in sorted(matches, key=lambda m: m.weight, reverse=True)]

        docs_matches = dict()
        for match in sorted(matches, key=lambda m: m.weight):
            if len(docs_matches) < config.retrieve_matches and match.doc_id not in docs_matches:
                docs_matches[match.doc_id] = match

        self.search_bar.update(1)

        return [str(k) for k in docs_matches.keys()]

    def _subset_heads(self) -> List[Tuple[int, int]]:
        from heads_idx import best_to_worst_heads
        return best_to_worst_heads
