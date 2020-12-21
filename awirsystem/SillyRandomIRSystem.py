from pv211_utils.irsystem import IRSystem
from pv211_utils.loader import load_documents
import random
from objects import Document


class SillyRandomIRSystem(IRSystem):
    """
    A system that returns documents in random order.
    """

    def __init__(self):
        documents = load_documents(Document)
        random_documents = list(documents.values())
        random.seed(21)
        random.shuffle(random_documents)
        self.random_documents = random_documents

    def search(self, query):
        """The ranked retrieval results for a query.

        Parameters
        ----------
        query : Query
            A query.

        Returns
        -------
        list of Document
            The ranked retrieval results for a query.

        """
        return self.random_documents