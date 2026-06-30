import numpy as np
import faiss


class Matcher:

    def __init__(self, students):

        self.students = students

        if len(students) == 0:
            self.index = None
            return

        embeddings = np.array(
            [s["embedding"] for s in students],
            dtype=np.float32
        )

        # normalize embeddings
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]

        # Inner Product == Cosine similarity after normalization using faiss
        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

    def match(self, query_embedding):

        if self.index is None:
            return None, -1

        query = np.array(
            [query_embedding],
            dtype=np.float32
        )

        faiss.normalize_L2(query)

        scores, indices = self.index.search(query, 1)

        score = float(scores[0][0])

        idx = int(indices[0][0])

        student = self.students[idx]

        return student, score