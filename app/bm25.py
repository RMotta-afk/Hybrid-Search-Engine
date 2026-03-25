import math
import re


class BM25Index:
    STOP_WORDS = {"the", "a", "an", "is", "are", "was", "were", "in", "on",
                  "at", "to", "for", "of", "and", "or", "but", "not", "with",
                  "this", "that", "it"}

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: dict = {}
        self.doc_lengths: dict = {}
        self.avg_doc_length: float = 0.0
        self.df: dict = {}
        self.total_docs: int = 0

    @staticmethod
    def tokenize(text: str) -> list[str]:
        tokens = re.findall(r'[a-z0-9]+', text.lower())
        return [t for t in tokens if t not in BM25Index.STOP_WORDS]

    def add_document(self, doc_id: str, text: str, metadata: dict = None):
        tokens = self.tokenize(text)
        tf: dict = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        self.documents[doc_id] = {"text": text, "tokens": tokens, "tf": tf, "metadata": metadata or {}}
        self.doc_lengths[doc_id] = len(tokens)
        self.total_docs += 1

        for term in tf:
            self.df[term] = self.df.get(term, 0) + 1

        self.avg_doc_length = sum(self.doc_lengths.values()) / self.total_docs

    def search(self, query: str, limit: int = 10) -> list[dict]:
        query_tokens = self.tokenize(query)
        scores: dict = {}

        for token in query_tokens:
            if token not in self.df:
                continue
            df = self.df[token]
            N = self.total_docs
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

            for doc_id, doc in self.documents.items():
                tf = doc["tf"].get(token, 0)
                if tf == 0:
                    continue
                doc_len = self.doc_lengths[doc_id]
                tf_score = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                )
                scores[doc_id] = scores.get(doc_id, 0) + idf * tf_score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [
            {
                "id": doc_id,
                "text": self.documents[doc_id]["text"],
                "metadata": self.documents[doc_id]["metadata"],
                "score": score,
            }
            for doc_id, score in ranked
        ]
