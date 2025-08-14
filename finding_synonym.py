import numpy as np
from sentence_transformers import SentenceTransformer

class SimpleVectorDB:
    def __init__(self):
        # Store original items (e.g., words or sentences)
        self.items = []
        # Store corresponding embedding vectors
        self.embeddings = []

    def add_item(self, item: str, embedding: np.ndarray):
        """
        Add a new item and its embedding vector to the database.
        :param item: The original word or sentence.
        :param embedding: The vector representation (numpy array).
        """
        self.items.append(item)
        self.embeddings.append(embedding)

    def find_similar(self, query_embedding, top_k: int = 3):
        """
        Find the most similar items to the given query embedding.
        Uses cosine similarity to compare vectors.

        :param query_embedding: The vector representation of the query.
        :param top_k: Number of similar results to return.
        :return: List of tuples (item, similarity_score).
        """
        # Return empty list if there are no embeddings in the database
        if not self.embeddings:
            return []

        # Calculate cosine similarity for each stored embedding
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) *  # Norm of each stored vector
            np.linalg.norm(query_embedding)            # Norm of query vector
        )

        # Get indices of top-k highest similarity scores (sorted in descending order)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Return the top-k items along with their similarity scores
        return [(self.items[i], similarities[i]) for i in top_indices]

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

db = SimpleVectorDB()
dictionary = ["cat", "dog", "fish","bird", "elephant", "king", "man"]

for item in dictionary:
    embedding = model.encode(item)
    db.add_item(item, embedding)

def search(query: str, model, db):
    # Encode the query to get its embedding
    query_embedding = model.encode(query)
    # Use the vector database to find similar items
    results = db.find_similar(query_embedding)
    print(f"Query: {query}")
    print("Similar items:")
    for item, score in results:
        print(f" - {item} (score: {score})")

search("Kitten", model, db)