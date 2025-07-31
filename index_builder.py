# Run this once to build index

import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

documents = [
    "AI is transforming the world.",
    "Machine learning enables systems to learn from data.",
    "Natural Language Processing deals with human language.",
    "Deep learning is a subset of machine learning.",
    "Ollama allows local LLMs to run easily on your machine.",
    "Vector databases like FAISS enable similarity search over embeddings."
]

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Save index and metadata
faiss.write_index(index, "faiss_index.index")
with open("doc_metadata.pkl", "wb") as f:
    pickle.dump(documents, f)

print("FAISS index and metadata saved.")
