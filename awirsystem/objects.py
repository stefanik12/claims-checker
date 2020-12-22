from pv211_utils.entities import QueryBase
from pv211_utils.entities import DocumentBase

import config


class Document(DocumentBase):
    """
    A Cranfield collection document.

    Parameters
    ----------
    document_id : int
        A unique identifier of the document.
    authors : list of str
        A unique identifiers of the authors of the document.
    bibliography : str
        The bibliographical entry for the document.
    title : str
        The title of the document.
    body : str
        The abstract of the document.

    """
    def __init__(self, document_id, authors, bibliography, title, body):
        super().__init__(document_id, authors, bibliography, title, body)
        # preprocessing?


class Query(QueryBase):
    """
    A Cranfield collection query.

    Parameters
    ----------
    query_id : int
        A unique identifier of the query.
    body : str
        The text of the query.

    """
    def __init__(self, query_id, body):
        super().__init__(query_id, body)
        # preprocessing!


class Match:

    def __init__(self, doc_id,  token: str, embedding_dist: float, attention: float):
        self.doc_id = doc_id
        self.token = token
        self.embedding_dist = embedding_dist
        self.attention = attention
        self.weight = self._eval_match()

    def _eval_match(self) -> float:
        return ((self.attention * config.weight_embedding_ratio) + self.embedding_dist) / \
               (config.weight_embedding_ratio + 1)

