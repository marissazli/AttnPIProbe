import math
import re
from collections import Counter


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text):
    return [tok.lower() for tok in TOKEN_PATTERN.findall(text)]


class Retriever:
    def retrieve(self, query, top_k=3):
        raise NotImplementedError


class LexicalRetriever(Retriever):
    """
    BM25-like lexical retriever over in-memory documents.
    """

    def __init__(self, corpus):
        self.corpus = corpus
        self.doc_tokens = []
        self.doc_term_freq = []
        self.doc_freq = Counter()
        self.avg_doc_len = 0.0

        for doc in corpus:
            tokens = _tokenize(doc["text"])
            tf = Counter(tokens)
            self.doc_tokens.append(tokens)
            self.doc_term_freq.append(tf)
            for term in tf.keys():
                self.doc_freq[term] += 1

        if self.doc_tokens:
            self.avg_doc_len = sum(len(toks) for toks in self.doc_tokens) / len(self.doc_tokens)

    def _idf(self, term):
        n_docs = len(self.corpus)
        df = self.doc_freq.get(term, 0)
        # BM25 idf with smoothing.
        return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

    def retrieve(self, query, top_k=3, k1=1.2, b=0.75):
        if top_k <= 0:
            return []

        query_terms = _tokenize(query)
        if not query_terms or not self.corpus:
            return []

        scores = []
        for idx, doc in enumerate(self.corpus):
            tf = self.doc_term_freq[idx]
            doc_len = len(self.doc_tokens[idx]) or 1

            score = 0.0
            for term in query_terms:
                if term not in tf:
                    continue
                term_tf = tf[term]
                idf = self._idf(term)
                denom = term_tf + k1 * (1 - b + b * (doc_len / (self.avg_doc_len or 1.0)))
                score += idf * ((term_tf * (k1 + 1)) / denom)

            if score > 0:
                scores.append(
                    {
                        "doc_id": doc["doc_id"],
                        "title": doc.get("title", ""),
                        "text": doc["text"],
                        "score": score,
                    }
                )

        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]

