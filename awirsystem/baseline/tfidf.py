# -*- coding: utf-8 -*-
"""PV211_Term_project_filippo.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FVJ2Jv5EKJhbKX5ZabsJOlwkni6YsUDk

# Term Project
“The Cranfield collection [...] was the pioneering test collection in allowing CRANFIELD precise quantitative measures of information retrieval effectiveness [...]. Collected in the United Kingdom starting in the late 1950s, it contains 1398 abstracts of aerodynamics journal articles, a set of 225 queries, and exhaustive relevance judgments of all (query, document) pairs.” [1, Section 8.2]

Your tasks, reviewed by your colleagues and the course instructors, are the following:

1.   *Implement a ranked retrieval system*, [1, Chapter 6] which will produce a list of documents from the Cranfield collection in a descending order of relevance to a query from the Cranfield collection. You MUST NOT use relevance judgements from the Cranfield collection in your information retrieval system. Relevance judgements MUST only be used for the evaluation of your information retrieval system.

2.   *Document your code* in accordance with [PEP 257](https://www.python.org/dev/peps/pep-0257/), ideally using [the NumPy style guide](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard) as seen in the code from exercises.
     *Stick to a consistent coding style* in accordance with [PEP 8](https://www.python.org/dev/peps/pep-0008/).

3.   *Reach at least 35% mean average precision* [1, Section 8.4] with your system on the Cranfield collection. You are encouraged to use techniques for tokenization, [1, Section 2.2] document representation [1, Section 6.4], tolerant retrieval [1, Chapter 3], relevance feedback and query expansion, [1, Chapter 9] and others discussed in the course.

4.   *Upload a link to your Google Colaboratory document to the homework vault in IS MU.* You MAY also include a brief description of your information retrieval system.

#### Install the fresh version of utils
"""

# ! pip
# install
# git + https: // gitlab.fi.muni.cz / xstefan3 / pv211 - utils.git @ master | grep
'^Successfully'

"""# Main idea of the Information Retrieval System

The idea behind this Information Retrieval System is to preprocess the documents and query, represent with the term-frequency inverse-document-frequency vectors and compute the cosine similarity to scoring the documents.

Details of the preprocessing and scoring will be provided in the next paragraphs.

There are more than one parameters that can be set to different values, the actual value reported is the value that maximizes the precision.
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

from collections import Counter


def remove_stop_words(words: list) -> list:
    """Remove the stop words from a list.

    Parameters
    ----------
    words : list
        List of string to be filtered from stop words

    Returns
    -------
    list of words
        List of input without the stop words.

    """

    stop_words = set(stopwords.words('english'))
    return [w for w in words if w not in stop_words]


def remove_punctuation(words: str) -> str:
    """Remove the stop words from a list.

    Parameters
    ----------
    words : str
        String to be filtered

    Returns
    -------
    str
        str of input without the punctuation.

    """

    punctuation = ("_", ",", ".", "\\", "/", "(", ")", "-", "?", "'")
    for c in punctuation:
        words = words.replace(c, " ")
    words = words.replace("  ", " ")
    return words


def stemming(words: list) -> list:
    """Apply stemming to a list of words.

    Parameters
    ----------
    words : list
        List of string to be stemmed

    Returns
    -------
    list of words
        List with the stemmed words.

    """

    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    return words


def k_gram(words: list, k=5) -> list:
    """Apply k-grams to a list of words.

    Parameters
    ----------
    words : list
        List of string
    k : int, optional
        Size of the k-gram (the default is 5)

    Returns
    -------
    list of words
        List of k-grams

    """

    temp = "$$".join(words)
    ris = []
    for i in range(len(temp) - k + 1):
        ris.append(temp[i:i + k])
    return ris


"""## Preprocessing
The preprocessing consists in the the following chain of operation:
 - remove puntuaction;
 - tokenize the words;
 - stemming the words;
 - appling k-grams with k=5;
 - adding to the result the words with length between 4 and 6.
"""


def preprocessing(words: str, k: int = 5, lower=4, upper=6) -> list:
    """Preprocess a text applying tokenization, stemming and k-grams.

    Parameters
    ----------
    words : str
        String to be preprocessed
    k : int, optional
        Size of the k-gram (the default is 5)
    lower, upper : int, optional
        The `lower` and `upper` size of the words to keep

    Returns
    -------
    list
        list of the preprocessed words

    """

    words = words.lower()
    words = remove_punctuation(words)

    words = word_tokenize(words)

    # words = remove_stop_words(words)
    # with stop word precision increase by 0.5%

    words = stemming(words)

    ris = k_gram(words, k)
    ris += [w for w in words if lower <= len(w) <= upper]
    # with words of defined length precision increase by almost 1%'''

    return ris


def term_frequency(words: list) -> dict:
    """Compute the log2 term frequency of a list of words.

    Parameters
    ----------
    words : list
        List of string

    Returns
    -------
    dict
        Dictionary with key the term and values the log2 frequency

    """

    tf = Counter(words)
    tf = {k: 1 + log2(v) for k, v in tf.items()}
    return tf


"""## Documents of Cranfield collection
Every document is represented as two vectors. The first one is the term frequency of the body, the authors, and the bibliography of the document. The second one is the term frequency of the title. In the code, different vector corresponds to a different category.
"""

from pv211_utils.entities import DocumentBase
from pv211_utils.loader import load_documents


class Document(DocumentBase):
    """A Cranfield collection document.

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

    Attributes
    ----------
    tf : dict
        Dictionary that contains for each category (as key) the corresponding
        dictionary of preprocess term frequency.

    """

    def __init__(self, document_id, authors, bibliography, title, body):
        super().__init__(document_id, authors, bibliography, title, body)

        self.tf = {}

        doc = {}
        doc["main"] = " ".join((body, authors, bibliography))
        doc["title"] = title

        category = ("main", "title")
        for cat in category:
            self.tf[cat] = term_frequency(preprocessing(doc[cat]))


"""## Implementation of your information retrieval system
The information retrieval system behaves in the following way.

In the constructor compute the normalize if-idf vector of each document.

In the search method:
 - perform the preprocessing of the query;
 - compute the if-idf of the query;
 - for each category compute the cosine distance;
 - sum the two distance with different weight;
 - ranked the documents according to the overall distance.

The cosine distance is computed by multiplying only the value of the term that appears both in query and document. It's not necessary to divide the result by the norm of the two vectors because:
 - the vector of the documents is already normalized;
 - the norm of the vector of the query is equal for all the distance, so it doesn't change the order.

"""

from collections import Counter

from math import log2
from math import sqrt

from pv211_utils.irsystem import IRSystem


class CharNGramTfIdfIRSystem(IRSystem):
    """
    A system that returns documents according to the distance from a query.

    Attributes
    ----------
    documents : list of Document
        Original documents.
    category: : tuple
        Each category to be compute.
    _weight : dict of weight
        Weight for each category compute.
    _result_size : int
        Size of the result to be returned.
    df : dict
        Dictionary that describes the document-frequency for each category.
    tf_idf : dict
        Dictionary that describes the tf-idf for each category for each
        document.
    N : int
        Number of documents.

    Parameters
    ----------
    documents : list of Document
         list of documents from the Cranfield collection.

    """

    def __init__(self):
        self.documents = load_documents(Document)
        self._weigth = {"main": 7, "title": 1}
        self._result_size = len(self.documents)
        self.category = ("main", "title")

        self.df = {}
        self.tf_idf = {}
        self.N = len(self.documents)

        for cat in self.category:

            self.df[cat] = {}
            for d_id, document in self.documents.items():
                for w in document.tf[cat].keys():
                    self.df[cat][w] = self.df[cat].get(w, 0) + 1

            self.tf_idf[cat] = {}

            # compute tf-idf for each document
            for d_id, document in self.documents.items():
                l = 0
                for w, co in document.tf[cat].items():
                    temp = co * log2(self.N / self.df[cat][w])
                    l += temp ** 2
                    if w not in self.tf_idf[cat]:
                        self.tf_idf[cat][w] = {}
                    self.tf_idf[cat][w][d_id] = temp

                # normalize the vector
                l = sqrt(l)
                for w in document.tf[cat].keys():
                    self.tf_idf[cat][w][d_id] /= l

    def set_weight(self, weigth: dict):
        """Set the weigth for the categories.

        Parameters
        ----------
        weigth : dict
            Dictionary of weight.

        """

        self._weigth = weigth

    def set_result_size(self, size: int):
        """Set the maximum result size.

        Parameters
        ----------
        size : int
            The maximum size.

        """

        self._result_size = size if size > 0 else len(self.documents)

    def search(self, query) -> list:
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

        words = preprocessing(query.body)
        tf = term_frequency(words)
        rank = {}

        for cat in self.category:
            tf_idf = {}

            # compute the tf-idf of the query
            for w, c in tf.items():
                if w in self.df[cat]:
                    temp = c * log2(self.N / self.df[cat][w])
                    tf_idf[w] = temp

            # compute the cosine similarity
            for w, q_val in tf_idf.items():
                for d_id, d_val in self.tf_idf[cat][w].items():
                    if d_id not in rank:
                        rank[d_id] = 0
                    rank[d_id] += (q_val * d_val * self._weigth[cat])

        ris = [(dist, d_id) for d_id, dist in rank.items() if dist > 0]
        ris.sort(reverse=True)
        ris = [self.documents[doc_id] for v, doc_id in ris]

        return ris[:self._result_size]


"""## Evaluation

The following code evaluates your information retrieval system using the Mean Average Precision evaluation measure.
"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# from pv211_utils.eval import mean_average_precision
# documents = load_documents(Document)
# IR = CharNGramTfIdfIRSystem(documents)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# mean_average_precision(IR, submit_result=False, author_name="Zanatta, Filippo")

"""# Additional note
Few operations increase the Mean Average Precision by 5-6%, but reduce performance in time, use more memory, make the IR more complex, and, maybe, commit overfitting.

The following list contains this operation:
 - using k-grams for the term;
 - keeping the words with a predetermined length;
 - not removing the stop word;
 - using two different distance and then made a weighted sum.

Also, returning all the documents with a distance greater than 0, and not only the first `k` or with some distance threshold, give a better result in term of precision, but a worst result from the point of view of the user.

## Bibliography
[1] Manning, Christopher D., Prabhakar Raghavan, and Hinrich Schütze. [*Introduction to information retrieval*](https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf). Cambridge university press, 2008.
"""