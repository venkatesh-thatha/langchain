from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

# Use a valid Hugging Face model (no dimensions parameter)
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

docs = [
    "The capital of India is New Delhi.",
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
    "The capital of Japan is Tokyo."
]

query = "Delhi is capital of India"

# Generate embeddings
doc_embeddings = embedding.embed_documents(docs)
query_embedding = embedding.embed_query(query)

# Compute cosine similarity
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# Find the most similar document
index = int(np.argmax(scores))
print("Most similar document:", docs[index])
print(f"Similarity Score: {scores[index]:.4f}")
