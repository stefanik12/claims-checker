from typing import Tuple, List

from pv211_utils.entities import QueryBase

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
                 weighting_device: str = "cpu", infer_device: str = "cpu",
                 firstn_heads=None, aggregation_strategy="per-layer-mean+sum"):
        # weight documents, index weighting and tokens by their embeddings, to use it then in search
        transformer = AutoTransformerModule(transformer_name_or_path, device=weighting_device)
        self.weighter = SelectedHeadsAttentionWeighting(transformer,
                                                        heads_subset=self._subset_heads(firstn_heads),
                                                        aggregation_strategy=aggregation_strategy)
        self.embeder = TransformerEmbedding(transformer, embed_strategy="last-4")
        self.inv_embedding_index = []
        embedding_idx = []
        self.doc_index = set()
        self.documents = load_documents(Document)
        for doc_id, doc in tqdm(list(self.documents.items()), desc="Indexing"):
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

    def search(self, query: QueryBase, weight_query: bool = False) -> List[Document]:
        """The ranked retrieval results for a query.

        Parameters
        ----------
            :param query : input query
            :param weight_query: Whether to weight matches of query tokens by their inner attention weights

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

        q_tokens, q_embeddings = self.embeder.tokenize_embed_text(query.body)
        q_tokens_w, q_weights = self.weighter.tokenize_weight_text(query.body)
        assert q_tokens == q_tokens_w

        norm = torch.linalg.norm(torch.tensor(q_embeddings), ord=2)
        q_embeddings_norm = torch.tensor(q_embeddings) / torch.max(norm, 1e-10 * torch.ones_like(norm))
        
        # similarity of query token tq to indexed token ti is tq_emb * ti_emb * ti_weight.T
        # ti_weight is contextualized attention of indexed token, using attention heads by selected methodology,
        # aggregated by selected methodology
        e_sims = torch.matmul(self.embedding_idx, q_embeddings_norm.T.to(self.embedding_idx.device))
        if weight_query:
            w_e_sims = e_sims * torch.tensor(q_weights, device=e_sims.device).expand_as(e_sims)
        else:
            w_e_sims = e_sims
            
        sims_sorted, idxs_sorted = torch.topk(w_e_sims, k=config.embedding_matches, largest=True, sorted=True, dim=0)

        for q_idx, q_token in enumerate(q_tokens):
            q_token_sims = sims_sorted[:, q_idx]
            q_token_idx = idxs_sorted[:, q_idx]
            for sim, (attention, match_token, match_doc_id) in zip(q_token_sims, [self.inv_embedding_index[idx]
                                                                                  for idx in q_token_idx]):
                matches.append(Match(match_doc_id, match_token, sim.item(), attention))

        docs_matches = {k: 0 for k in set([m.doc_id for m in matches])}
        for match in matches:
            docs_matches[match.doc_id] += match.weight

        top_docs = [doc_id for doc_id in sorted(list(docs_matches.keys()), key=lambda k: docs_matches[k], reverse=True)]

        self.search_bar.update(1)

        return [self.documents[k] for k in top_docs[:config.retrieve_matches]]

    @staticmethod
    def _subset_heads(firstn_heads: int = None) -> List[Tuple[int, int]]:
        from heads_idx import best_to_worst_heads
        if firstn_heads is None:
            firstn_heads = len(best_to_worst_heads)
        print("Using top %s heads" % firstn_heads)
        return best_to_worst_heads[:firstn_heads]
